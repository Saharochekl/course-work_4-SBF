# mast_rest.py
import json, os, requests

BASE = "https://mast.stsci.edu/api/v0/invoke"

def mast_query(payload: dict) -> dict:
    hdr = {"Content-type": "application/x-www-form-urlencoded","Accept":"text/plain"}
    s = json.dumps(payload)
    r = requests.post(f"{BASE}", data="request="+requests.utils.quote(s), headers=hdr)
    r.raise_for_status()
    return r.json()

# 1) Name resolver
resolved = mast_query({"service":"Mast.Name.Lookup","params":{"input":"NGC 5584","format":"json"}})
ra  = resolved["resolvedCoordinate"][0]["ra"]
dec = resolved["resolvedCoordinate"][0]["decl"]

# 2) Отобрать наблюдения JWST вокруг координат и по фильтрам
filters = [
  {"paramName":"obs_collection","values":["JWST"]},
  {"paramName":"instrument_name","values":["JWST/NIRCam","JWST/NIRISS","JWST/MIRI"]},
  {"paramName":"filters","values":["F090W","F150W","F200W","F356W"], "separator":";"},
]
obs = mast_query({
  "service":"Mast.Caom.Filtered.Position",
  "format":"json",
  "params":{"columns":"*","filters":filters,"position":f"{ra}, {dec}, 0.1"}
})

# 3) Пройтись по obsid → продукты, отфильтровать SCIENCE и *_i2d.fits
products = []
for row in obs["data"]:
    pr = mast_query({"service":"Mast.Caom.Products","format":"json","params":{"obsid":row["obsid"]}})
    for x in pr["data"]:
        if x.get("productType")=="SCIENCE" and x["productFilename"].endswith("_i2d.fits"):
            products.append(x)

print("К закачке:", len(products))
# 4) Скачивание (по одному)
for p in products:
    url = "https://mast.stsci.edu/api/v0.1/Download/file"
    outdir = os.path.join("mastFiles", p["obs_collection"], p["obs_id"])
    os.makedirs(outdir, exist_ok=True)
    resp = requests.get(url, params={"uri": p["dataURI"]})
    with open(os.path.join(outdir, os.path.basename(p["productFilename"])), "wb") as f:
        f.write(resp.content)