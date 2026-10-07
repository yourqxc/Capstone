"""고정한 새 입력 4쌍을 다시 실행한다. ADE20K 로컬 데이터가 필요하다.
선정 조건과 이 실행에서 발견한 실패는 DEVLOG §36에 기록한다.
"""
import sys,json,time
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageOps
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
import app
from pipeline.segment import segment,cutout
ade=Path.home()/'capstone-data/ADEChallengeData2016';selection=json.loads((root/'docs/evidence/38_new_inputs/selection.json').read_text());out=root/'docs/evidence/38_new_inputs';out.mkdir(exist_ok=True)
records=[];sheet=Image.new('RGB',(1500,len(selection)*390),'white');draw=ImageDraw.Draw(sheet)
for i,s in enumerate(selection):
 iid=s['id'];source=Image.open(ade/'images/validation'/f'{iid}.jpg').convert('RGB');ann=np.array(Image.open(ade/'annotations/validation'/f'{iid}.png'))
 _,lab,_,_=cv2.connectedComponentsWithStats(((ann==20)|(ann==31)).astype('uint8'),8)
 x,y,w,h=s['bbox'];pad=max(20,int(max(w,h)*.25));bounds=(max(0,x-pad),max(0,y-pad),min(source.width,x+w+pad),min(source.height,y+h+pad));item=source.crop(bounds);box=(x-bounds[0],y-bounds[1],x+w-bounds[0],y+h-bounds[1]);gt=(lab==s['component'])[bounds[1]:bounds[3],bounds[0]:bounds[2]]
 start=time.perf_counter();mask=segment(item,box);before=cutout(item,box,mask=mask,refine_edges=False,crop=False);after=cutout(item,box,mask=mask,crop=False)
 scores={}
 for key,obj in [('sam',before),('refined',after)]:
  predicted=np.array(obj.getchannel('A'))>127;scores[key+'_iou']=round(float((predicted&gt).sum()/max((predicted|gt).sum(),1)),4)
 room_id=selection[(i+1)%len(selection)]['id'];room=Image.open(ade/'images/validation'/f'{room_id}.jpg').convert('RGB')
 # Pass the actual refined alpha to the app, so its segmentation is not recomputed.
 result=app.run({'background':room},{'background':after},'new chair','로컬','세워놓기','자동','수동',1.,False,35,55,30,35,progress=lambda *a,**k:None)
 item.save(out/f'{i+1}_source.png');after.save(out/f'{i+1}_cutout.png');result[2].save(out/f'{i+1}_result.png');(out/f'{i+1}_log.txt').write_text(result[3])
 for col,(label,im) in enumerate([('Source (new)',item),('Cutout (new)',after),('Composite (new room)',result[2])]):
  bg=Image.new('RGBA',im.size,'#8d8d8d');bg.alpha_composite(im.convert('RGBA'));thumb=ImageOps.contain(bg.convert('RGB'),(490,350));sheet.paste(thumb,(col*500+(490-thumb.width)//2,i*390+30));draw.text((col*500+10,i*390+8),f'{i+1} {label}',fill='black')
 record={**s,'room_id':room_id,**scores,'refinement':after.info['edge_refinement'],'seconds':round(time.perf_counter()-start,2),'log':result[3]};records.append(record);print(record,flush=True)
(out/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));sheet.save(root/'docs/evidence/38_새사진_전체결과.png')
