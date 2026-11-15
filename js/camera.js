// Camera access via getUserMedia. Works with phone cameras, laptop webcams and USB/industrial
// cameras that the operating system exposes as a webcam (UVC).
export class Camera {
  constructor(video) {
    this.video = video;
    this.stream = null;
    this.facing = 'environment';
    this.torchOn = false;
  }

  get active() { return !!this.stream; }

  static supported() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  async list() {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'videoinput');
  }

  async start(deviceId) {
    this.stop();
    const size = { width: { ideal: 1280 }, height: { ideal: 960 } };
    const video = deviceId ? { deviceId: { exact: deviceId }, ...size } : { facingMode: { ideal: this.facing }, ...size };
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video });
    this.video.srcObject = this.stream;
    await new Promise((r) => {
      if (this.video.readyState >= 1) r();
      else this.video.onloadedmetadata = () => r();
    });
    await this.video.play();
    const track = this.stream.getVideoTracks()[0];
    this.torchOn = false;
    return { label: track.label, deviceId: track.getSettings().deviceId };
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  hasTorch() {
    const track = this.stream?.getVideoTracks()[0];
    const caps = track?.getCapabilities ? track.getCapabilities() : {};
    return !!caps.torch;
  }

  async toggleTorch() {
    const track = this.stream?.getVideoTracks()[0];
    if (!track) return false;
    this.torchOn = !this.torchOn;
    await track.applyConstraints({ advanced: [{ torch: this.torchOn }] });
    return this.torchOn;
  }

  get width() { return this.video.videoWidth; }
  get height() { return this.video.videoHeight; }
}

// Region of interest (fractions of the frame) for the given settings and frame size.
export function roiFor(settings, W, H) {
  const s = settings.roiScale;
  if (settings.rows === 1 && settings.cols === 1) {
    const side = s * Math.min(W, H);
    const w = side / W, h = side / H;
    return { x: (1 - w) / 2, y: (1 - h) / 2, w, h };
  }
  return { x: (1 - s) / 2, y: (1 - s) / 2, w: s, h: s };
}

// Crop the ROI out of any drawable source (video, image, canvas) and split it into grid cells.
export function grabCells(source, W, H, settings) {
  const roi = roiFor(settings, W, H);
  const { rows, cols } = settings;
  const sx = roi.x * W, sy = roi.y * H, sw = roi.w * W, sh = roi.h * H;
  const cw = sw / cols, ch = sh / rows;
  const cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cv = document.createElement('canvas');
      cv.width = 224; cv.height = 224;
      const ctx = cv.getContext('2d');
      ctx.drawImage(source, sx + c * cw, sy + r * ch, cw, ch, 0, 0, 224, 224);
      cells.push({ canvas: cv, row: r, col: c });
    }
  }
  return cells;
}

export function drawOverlay(canvas, settings, W, H, results) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!W || !H) return;
  const roi = roiFor(settings, W, H);
  const X = roi.x * canvas.width, Y = roi.y * canvas.height, Wd = roi.w * canvas.width, Hd = roi.h * canvas.height;
  // dim outside
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.fillRect(0, 0, canvas.width, Y);
  ctx.fillRect(0, Y + Hd, canvas.width, canvas.height - Y - Hd);
  ctx.fillRect(0, Y, X, Hd);
  ctx.fillRect(X + Wd, Y, canvas.width - X - Wd, Hd);
  const cw = Wd / settings.cols, ch = Hd / settings.rows;
  const colors = { pass: '#22c55e', review: '#f5a524', reject: '#ef4444' };
  ctx.lineWidth = 2 * dpr;
  for (let r = 0; r < settings.rows; r++) {
    for (let c = 0; c < settings.cols; c++) {
      const res = results?.find((x) => x.row === r && x.col === c);
      ctx.strokeStyle = res ? colors[res.verdict] : 'rgba(245,165,36,0.9)';
      ctx.lineWidth = (res ? 4 : 2) * dpr;
      ctx.strokeRect(X + c * cw + dpr, Y + r * ch + dpr, cw - 2 * dpr, ch - 2 * dpr);
      if (res) {
        ctx.fillStyle = colors[res.verdict];
        const label = res.verdict.toUpperCase();
        ctx.font = `bold ${Math.max(10, Math.min(18, cw / 7)) * dpr}px system-ui`;
        const tw = ctx.measureText(label).width + 8 * dpr;
        ctx.fillRect(X + c * cw + 2 * dpr, Y + r * ch + 2 * dpr, tw, 20 * dpr);
        ctx.fillStyle = res.verdict === 'reject' ? '#fff' : '#111';
        ctx.fillText(label, X + c * cw + 6 * dpr, Y + r * ch + 17 * dpr);
      }
    }
  }
}
