"""End-to-end test of QualiEdge AI Inspector in headless Chromium with a fake camera.

Usage:
  python tests/make_test_images.py
  python -m http.server 8765 --directory app &
  python tests/e2e.py [URL]
"""
import sys, time, json
from playwright.sync_api import sync_playwright
BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8765/'
import os
I = os.environ.get('QE_IMGS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imgs')) + '/'
errors = []
answers = []
with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--no-sandbox'])
    ctx = b.new_context(viewport={'width': 412, 'height': 900}, permissions=['camera'], accept_downloads=True)
    pg = ctx.new_page()
    pg.on('console', lambda m: m.type in ('error', 'warning') and errors.append(f'{m.type}: {m.text}'))
    pg.on('pageerror', lambda e: errors.append(f'pageerror: {e}'))
    pg.on('dialog', lambda d: d.accept(answers.pop(0)) if d.type == 'prompt' else d.accept())
    pg.goto(BASE)
    pg.wait_for_function("window.qualiedge && !document.getElementById('modelPill').textContent.includes('Loading')", timeout=60000)
    print('engine:', pg.inner_text('#engineInfo'))
    pg.click('#camStartBtn')
    pg.wait_for_function("document.getElementById('video').videoWidth > 0", timeout=15000)
    print('camera:', pg.eval_on_selector('#video', 'v => v.videoWidth + "x" + v.videoHeight'), pg.eval_on_selector('#camSelect', 's => s.options.length'), 'cams')
    pg.screenshot(path='/tmp/qe-s1_inspect_empty.png')
    pg.click('#nav-train')
    pg.click('#addGoodBtn'); pg.wait_for_function("document.querySelectorAll('#gallery .thumb').length == 1", timeout=20000)
    print('camera capture ->', pg.inner_text('#sampleCounts'))
    pg.click('#gallery .thumb'); pg.click('#sampleDialog [data-act=delete]')
    pg.wait_for_function("document.querySelectorAll('#gallery .thumb').length == 0")
    pg.set_input_files('#uploadGood', [f'{I}good_{i}.png' for i in range(14)])
    pg.wait_for_function("document.querySelectorAll('#gallery .thumb').length == 14", timeout=60000)
    answers.append('missing clip'); pg.click('#newTypeBtn')
    pg.set_input_files('#uploadDefect', [f'{I}missing_{i}.png' for i in range(5)])
    pg.wait_for_function("document.querySelectorAll('#gallery .thumb').length == 19", timeout=60000)
    answers.append('scratch'); pg.click('#newTypeBtn')
    pg.set_input_files('#uploadDefect', [f'{I}scratch_{i}.png' for i in range(5)])
    pg.wait_for_function("document.querySelectorAll('#gallery .thumb').length == 24", timeout=60000)
    print('samples:', pg.inner_text('#sampleCounts'))
    t = time.time(); pg.click('#trainBtn')
    pg.wait_for_function("document.getElementById('modelPill').textContent.startsWith('Ready')", timeout=180000)
    print(f'trained in {time.time()-t:.1f}s:', pg.inner_text('#trainStatus'), '|', pg.inner_text('#modelPill'))
    pg.click('#evalBtn'); pg.wait_for_selector('#evalResult:not([hidden])', timeout=180000)
    print('EVAL:', pg.inner_text('#evalResult').replace('\n', ' | '))
    pg.screenshot(path='/tmp/qe-s2_train.png', full_page=True)
    pg.click('#nav-inspect')
    def insp(f):
        pg.evaluate("document.getElementById('resultCard').hidden = true")
        pg.set_input_files('#inspectFile', I + f)
        pg.wait_for_selector('#resultCard:not([hidden])', timeout=30000)
        return pg.inner_text('#resultVerdict'), pg.inner_text('#resultMeta')
    pg.fill('#lotInput', 'LOT-TEST-1'); pg.click('#batchToggleBtn')
    for f in ['test_good_0.png', 'test_good_1.png', 'test_good_2.png', 'test_good_3.png', 'test_missing.png', 'test_scratch.png']:
        print(f, '->', insp(f))
    print('batch bar:', pg.inner_text('#batchBar').replace('\n', ' '))
    pg.click('#inspectBtn'); pg.wait_for_timeout(1500)
    print('camera inspect ->', pg.inner_text('#resultVerdict'), pg.inner_text('#resultMeta'))
    pg.click('#batchToggleBtn')
    # tray mode
    pg.click('#nav-settings'); pg.fill('#setRows', '2'); pg.dispatch_event('#setRows', 'change'); pg.wait_for_timeout(400); pg.fill('#setCols', '2'); pg.dispatch_event('#setCols', 'change'); pg.wait_for_timeout(400)
    pg.click('#nav-inspect')
    pg.fill('#lotInput', 'TRAY-7'); pg.click('#batchToggleBtn')
    print('tray one defect ->', insp('tray_one_defect.png'), [c.inner_text().replace('\n',' ') for c in pg.query_selector_all('#resultCells .cell')])
    print('tray all good ->', insp('tray_all_good.png'))
    print('tray batch:', pg.inner_text('#batchBar').replace('\n', ' '))
    pg.screenshot(path='/tmp/qe-s3_inspect.png', full_page=True)
    pg.click('#nav-batches'); pg.wait_for_timeout(800)
    print('batches:', [x.inner_text().replace('\n', ' ') for x in pg.query_selector_all('.batch-item')])
    pg.click('.batch-item')
    with pg.expect_download() as dl: pg.click('text=CSV')
    path = dl.value.path(); print('CSV:', open(path).read().splitlines()[:3])
    # operator review: mark first review/reject as good
    btns = pg.query_selector_all('#batchDetail .review-actions .pass-btn'); btns[-1].click(); pg.wait_for_timeout(800)
    print('after override:', pg.inner_text('#batchDetail .batch-verdict'))
    pg.screenshot(path='/tmp/qe-s4_batch.png', full_page=True)
    # persistence
    pg.reload()
    pg.wait_for_function("window.qualiedge && !document.getElementById('modelPill').textContent.includes('Loading')", timeout=60000)
    print('after reload pill:', pg.inner_text('#modelPill'))
    print('reload inspect ->', insp('test_missing.png'))
    # export
    pg.click('#nav-settings')
    with pg.expect_download() as dl: pg.click('#exportRecipeBtn')
    d = json.load(open(dl.value.path())); print('export:', d['format'], len(d['samples']), 'samples')
    pg.screenshot(path='/tmp/qe-s5_settings.png', full_page=True)
    b.close()
print('ERRORS:', errors or 'none')
