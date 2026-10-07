"""§36의 고정 입력을 현재 자동 배치로 재실행. 이전 결과와 나란히 저장한다."""
import sys,json,time
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root));import app
ade=Path.home()/'capstone-data/ADEChallengeData2016';source=root/'docs/evidence/38_new_inputs';out=root/'docs/evidence/40_placement';out.mkdir(exist_ok=True)
records=json.loads((source/'records.json').read_text());sheet=Image.new('RGB',(1200,len(records)*430),'white');d=ImageDraw.Draw(sheet)
for i,r in enumerate(records):
 room=Image.open(ade/'images/validation'/f'{r["room_id"]}.jpg').convert('RGB');item=Image.open(source/f'{i+1}_cutout.png');t=time.perf_counter()
 result=app.run({'background':room},{'background':item},'chair','로컬','세워놓기','자동','수동',1.,False,35,55,30,35,progress=lambda *a,**k:None)
 result[2].save(out/f'{i+1}_result.png');(out/f'{i+1}_log.txt').write_text(result[3]);print(i+1,result[3],flush=True)
 for k,im in enumerate([Image.open(source/f'{i+1}_result.png'),result[2]]):
  d.text((k*600+8,i*430+5),f'{i+1} '+('Before' if k==0 else 'After floor placement'),fill='black');thumb=ImageOps.contain(im,(595,395));sheet.paste(thumb,(k*600,i*430+28))
sheet.save(root/'docs/evidence/40_바닥배치_전후.png')
