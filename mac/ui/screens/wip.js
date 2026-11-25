// Side menus of the 이미지 and 영상 생성 tabs (design §11)
import { t, html } from '../app.js';

export const IMAGE_MENU = () => [
  { items: [['#/image/gen', t('side.gen')], ['#/image/edit', t('side.edit')]] },
  { title: t('side.workflow'), items: [['#/image/character', t('side.character')], ['#/image/location', t('side.location')], ['#/image/prop', t('side.prop')], ['#/image/scene', t('side.scene')]] },
];
export const VIDEO_MENU = () => [
  { items: [['#/video/one', t('side.one')], ['#/video/finish', t('side.finish')]] },
  { title: t('side.workflow'), items: [['#/video/script', t('side.script')], ['#/video/seq', t('side.seq')], ['#/video/film', t('side.film')]] },
];

export function sideMenu(groups, current) {
  return html`<nav class="side" aria-label="${t('side.menu')}">${groups.map((g) => html`
    ${g.title ? html`<div class="grp">${g.title}</div>` : ''}
    ${g.items.map(([href, label]) => html`<a href="${href}" aria-current="${href === current ? 'page' : 'false'}">${label}</a>`)}`)}</nav>`;
}

