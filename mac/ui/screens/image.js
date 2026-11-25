// 이미지: 생성 · 편집 (imagegen.js) / 워크플로 캐릭터 (character.js) · 배경 · 사물 (location.js) · 장면 (scene.js)
const SUB = { gen: './imagegen.js', edit: './imagegen.js', character: './character.js', location: './location.js', prop: './location.js', scene: './scene.js' };
let inner = null;
export default {
  async mount(el, ctx) {
    inner = (await import(SUB[ctx.parts[1]] || SUB.character)).default;
    return inner.mount(el, ctx);
  },
  update() { if (inner && inner.update) return inner.update(); },
  unmount() { if (inner && inner.unmount) inner.unmount(); inner = null; },
};
