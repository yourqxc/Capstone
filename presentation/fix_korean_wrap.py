"""pptxgenjs 결과에 한글 단어 잘림 방지(eaLnBrk=0)를 모든 문단에 넣는다."""
import sys, zipfile, shutil
from lxml import etree
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
src, dst = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src); zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
n = 0
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
        root = etree.fromstring(data)
        for p in root.iter(f"{{{A}}}p"):
            ppr = p.find(f"{{{A}}}pPr")
            if ppr is None:
                ppr = etree.Element(f"{{{A}}}pPr"); p.insert(0, ppr)
            ppr.set("eaLnBrk", "0"); n += 1
        data = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    zout.writestr(item, data)
zout.close(); print("문단", n, "개에 eaLnBrk=0")
