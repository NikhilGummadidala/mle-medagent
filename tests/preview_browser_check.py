"""Browser component checks using real input images; no agent responses are mocked."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parents[1]
# Synthetic, non-medical component fixtures. These never enter agent inference
# and are not used as patient evaluation images or screenshots.
from io import BytesIO
from PIL import Image

def png_fixture(size,color):
    buffer=BytesIO()
    Image.new('RGB',size,color).save(buffer,format='PNG')
    return buffer.getvalue()

first=png_fixture((320,160),(40,90,150))
second=png_fixture((160,320),(130,70,40))
with sync_playwright() as pw:
    browser=pw.chromium.launch()
    page=browser.new_page()
    page.goto('http://127.0.0.1:8000/')
    page.locator('#fileInput').set_input_files([
      {'name':'mammogram_" onload="window.__previewInjected=true.png','mimeType':'image/png','buffer':first},
      {'name':'mammogram_second.png','mimeType':'image/png','buffer':second}])
    result=page.evaluate('''async () => {
      const attachments=await Promise.all(_files.map(_imagePreview));
      const host=document.createElement('div');
      host.innerHTML=_msgHTML({role:'user',content:'Preview component regression fixture',attachments});
      document.body.appendChild(host);
      const imgs=[...host.querySelectorAll('img')];
      await Promise.all(imgs.map(img=>img.decode()));
      const values={count:imgs.length,distinct:attachments[0].preview!==attachments[1].preview,
        bounded:imgs.every(img=>Math.max(img.naturalWidth,img.naturalHeight)<=240),
        aspectRatioPreserved:imgs[0].naturalWidth/imgs[0].naturalHeight===2&&imgs[1].naturalWidth/imgs[1].naturalHeight===.5,
        noInjectedHandler:imgs.every(img=>!img.hasAttribute('onload'))&&!window.__previewInjected,
        names:[...host.querySelectorAll('figcaption')].map(el=>el.textContent),
        legacyCompatible:!_msgHTML({role:'user',content:'An older text message'}).includes('<img'),
        unsafeSourceRejected:!_attachmentHTML({role:'user',attachments:[{name:'test',preview:'javascript:alert(1)'}]}).includes('<img')};
      const corrupt=await _imagePreview(new File(['not an image'],'invalid.png',{type:'image/png'}));
      values.invalidImageHasFallback=!corrupt.preview&&corrupt.name==='invalid.png';
      const text=await _imagePreview(new File(['text'],'note.txt',{type:'text/plain'}));
      values.nonImageHasFallback=!text.preview&&text.name==='note.txt';
      host.remove();return values;
    }''')
    assert result['count']==2 and result['distinct'] and result['bounded'] and result['aspectRatioPreserved']
    for key in ['noInjectedHandler','legacyCompatible','unsafeSourceRejected','invalidImageHasFallback','nonImageHasFallback']:assert result[key],key
    assert result['names'][0]=='mammogram_" onload="window.__previewInjected=true.png'
    print(json.dumps(result))
    browser.close()
