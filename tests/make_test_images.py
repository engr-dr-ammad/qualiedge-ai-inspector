"""Synthetic test parts for the end-to-end test: a plate with a slot and a blue clip.
Defects: missing clip, scratch. Writes to tests/imgs/."""
import os, random
from PIL import Image, ImageDraw, ImageEnhance

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imgs')
os.makedirs(OUT, exist_ok=True)
random.seed(1)

def part(defect=None, size=400):
    im = Image.new('RGB', (size, size), (70, 75, 80)); d = ImageDraw.Draw(im)
    ox, oy = random.randint(-10, 10), random.randint(-10, 10)
    d.rounded_rectangle((80 + ox, 80 + oy, 320 + ox, 320 + oy), radius=20, fill=(200, 200, 195))
    d.rectangle((120 + ox, 180 + oy, 280 + ox, 220 + oy), fill=(40, 40, 45))
    if defect != 'missing': d.ellipse((240 + ox, 100 + oy, 290 + ox, 150 + oy), fill=(30, 110, 200))
    if defect == 'scratch': d.line((100 + ox, 280 + oy, 300 + ox, 250 + oy), fill=(90, 40, 30), width=8)
    return ImageEnhance.Brightness(im).enhance(random.uniform(.85, 1.15))

save = lambda im, n: im.save(os.path.join(OUT, n))
for i in range(14): save(part(), f'good_{i}.png')
for i in range(5): save(part('missing'), f'missing_{i}.png')
for i in range(5): save(part('scratch'), f'scratch_{i}.png')
for i in range(4): save(part(), f'test_good_{i}.png')
save(part('missing'), 'test_missing.png'); save(part('scratch'), 'test_scratch.png')
for name, bad in (('tray_one_defect.png', 3), ('tray_all_good.png', None)):
    tray = Image.new('RGB', (800, 800))
    for k, (x, y) in enumerate([(0, 0), (400, 0), (0, 400), (400, 400)]):
        tray.paste(part('missing' if k == bad else None), (x, y))
    save(tray, name)
print('test images written to', OUT)
