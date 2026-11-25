// 영상 생성: 한 컷 (videogen.js) · 마무리 (finish.js) / 워크플로 시퀀스 (sequence.js) · 한 편 (film.js) · 대본 (script.js)
let inner = null;
export default {
  async mount(el, ctx) {
    inner = (await import({ seq: './sequence.js', finish: './finish.js', film: './film.js', script: './script.js' }[ctx.parts[1]] || './videogen.js')).default;
    return inner.mount(el, ctx);
  },
  update() { if (inner && inner.update) return inner.update(); },
  unmount() { if (inner && inner.unmount) inner.unmount(); inner = null; },
};
