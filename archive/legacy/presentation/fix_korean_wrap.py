"""pptxgenjs 한글 줄바꿈과 존재하지 않는 마스터 선언을 표준 라이브러리로 정리한다."""
import sys
import zipfile
import xml.etree.ElementTree as etree
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
etree.register_namespace("", "http://schemas.openxmlformats.org/package/2006/content-types")
src, dst = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src); zout = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED)
parts = set(zin.namelist())
n = 0
for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename == "[Content_Types].xml":
        root = etree.fromstring(data)
        # pptxgenjs 4.0.1은 파일이 없는 slideMaster2...도 형식 목록에 선언한다.
        for entry in list(root):
            target = entry.get("PartName")
            if target is not None and target.lstrip("/") not in parts:
                root.remove(entry)
        data = etree.tostring(root, xml_declaration=True, encoding="UTF-8")
    elif item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
        root = etree.fromstring(data)
        for p in root.iter(f"{{{A}}}p"):
            ppr = p.find(f"{{{A}}}pPr")
            if ppr is None:
                ppr = etree.Element(f"{{{A}}}pPr"); p.insert(0, ppr)
            ppr.set("eaLnBrk", "0"); n += 1
        data = etree.tostring(root, xml_declaration=True, encoding="UTF-8")
    zout.writestr(item, data)
zin.close(); zout.close(); print("문단", n, "개에 eaLnBrk=0")
