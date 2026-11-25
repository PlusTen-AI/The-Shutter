"""Prompt templates and default settings for character/place jobs (production doc ch. 2·3 rules). The server fills them
in when it splits a job into units.

Job options carry only the 'what'. To use a whole prompt instead of a template, pass options.prompt.
"""
import re


# 2.1 master rules: PE off, 25 steps, 4MP, 2:3, light grey background, front medium shot (production ch. 2)
SKIN = ("Skin: realistic unretouched skin with visible fine pores on the nose and cheeks, faint peach fuzz along the jaw "
        "and cheeks, subtle natural unevenness in tone, slight natural redness on the cheeks, knuckles and elbows, faint "
        "texture under the eyes, a soft natural sheen on the forehead and nose bridge, one continuous skin tone and white "
        "balance across face, neck, collarbone, arms and hands, subtle elegant makeup, soft rosy nude lips, not airbrushed, "
        "no beauty filter, no skin smoothing.")
SKIN_TURBO = ("Skin: clear, even, youthful complexion with a natural fine texture up close, very fine pores visible only on "
              "the nose, a soft healthy glow on the forehead and nose bridge, one continuous skin tone and white balance "
              "across face, neck, collarbone, arms and hands, smooth youthful hands and arms, subtle elegant makeup, soft "
              "rosy nude lips, natural skin that is not airbrushed.")
# Master framing = the FL02 master wording. Naming shoes in the master's clothing made Qwen fit the whole body head to
# shoes into the 2:3 frame and squeeze it (FL02/ML02 at first, MS01_ref_v1 in drama test 02: legs 0.37 of nose→ankle vs
# 0.50-0.56 on good full-body panels) — so the master's clothing drops footwear (_no_footwear); the card keeps the full
# outfit for the sheet's full-body panels. The worker checks both (pose: ankles on a master → redraw; legs < 0.45 on a
# full-body panel → redraw).
FRAMING = ("Framing: waist-up medium shot, the frame ends just below the hips, front view, standing straight facing the "
           "camera, arms relaxed at {pos} sides, calm neutral expression, looking into the lens. Natural adult proportions.")
FOOTWEAR = re.compile(r'\b(shoes?|sneakers?|trainers?|heels?|high-heeled|pumps|loafers?|boots?|sandals?|stilettos?|slippers?|'
                      r'flats|mules|slingbacks?|socks?|stockings?|tights)\b'
                      # Korean card text (UI, session #10: a Korean card kept '스니커즈' and the master came out full-body)
                      r'|신발|운동화|스니커즈|구두|힐|부츠|샌들|슬리퍼|로퍼|펌프스|스틸레토|뮬|양말|스타킹', re.I)


FULL_FRAME = re.compile(r'full[ -]length|feet|shoes|toes?\b|whole body|전신|발끝', re.I)   # a frame that shows the feet


def _no_footwear(clothing):
    """The card's clothing sentence without its footwear items (comma / semicolon separated) — for the waist-up master."""
    parts = re.split(r'(?<=[,;])\s+', clothing.strip().rstrip('.'))
    keep = [p for p in parts if not FOOTWEAR.search(p)]
    return ' '.join(keep).rstrip(',;') + '.' if keep else ''
BACKGROUND = ("Background and light: plain light grey seamless studio backdrop, soft key light from the front left, gentle "
              "fill from the right, natural shadow under the chin, true-to-life colour, sharp focus on the eyes.")
FACE_LOCK = ("Keep the {who} from image 1 as exactly the same person. The face must match image 1 precisely: same eye shape "
             "and double eyelids, same eyebrows, same nose bridge and tip, same lip shape, same jawline and chin, same skin "
             "tone, same hair colour and length.")
FACE_LOCK_END = ("The face is sharp with natural skin texture and fine pores. {Pron} must be instantly recognizable as the "
                 "{who} in image 1.")
DERIVE = {   # kind → scene sentence (angles near the source first, small expressions — production ch. 2 'into a scene')
    'angle_34L': "Same studio, light and outfit. {Pron} turns {pos} body and face about 45 degrees to {pos} own left, which is the right side of the image, three-quarter view, calm neutral expression.",
    'angle_34R': "Same studio, light and outfit. {Pron} turns {pos} body and face about 45 degrees to {pos} own right, which is the left side of the image, three-quarter view, calm neutral expression.",
    'side_L': "Same studio, light and outfit. Full profile view facing {pos} own left, which is the right side of the image, calm neutral expression.",
    'full': "Same studio, light and outfit. Full-body shot from head to shoes, standing upright, arms relaxed at {pos} sides, the whole body visible with space above the head and below the feet.",
    'closeup': "Same studio and light. Head-and-shoulders close-up, front view, calm neutral expression, looking into the lens.",
    'smile': "Same studio, light, outfit and framing. A soft closed-mouth smile, everything else unchanged.",
    'talk': "Same studio, light, outfit and framing. Mouth slightly open as if speaking, neutral expression, everything else unchanged.",
}
PHOTO_TO_REF = ("Turn the {who} in image 1 into a clean character reference photo of exactly the same person: same face, same "
                "eye shape, eyebrows, nose, lips, jawline, face shape, hair colour and length, same skin tone. "
                "{clothing} " + FRAMING + " " + BACKGROUND + " Keep realistic skin texture with fine pores, no airbrushing. "
                "{Pron} must be instantly recognizable as the {who} in image 1.")
PLACE = ("Wide establishing {kind} photo of {space}, no people. {scale}\n{body}\n"
         "Camera: eye level at standing height, 18mm wide lens, f/8, everything in sharp focus, straight vertical lines, raw photo.")
PLACE_DERIVE = ("The same building as image 1: keep exactly the same floor, wall and ceiling materials, the same lighting, "
                "colour tone and time of day. {view} Only the elements of this space are in view, no people.")


def _pron(o):
    f = str(o.get('gender', 'f')).lower().startswith('f')
    kid = o.get('child') in (True, 1, '1', 'true', 'True')   # 기능 13: a child is a girl / boy in every template
    return {'who': o.get('who') or (('girl' if kid else 'woman') if f else ('boy' if kid else 'man')), 'Pron': 'She' if f else 'He',
            'pron': 'she' if f else 'he', 'pos': 'her' if f else 'his'}


def char_ref(o):
    """Text → master. o: who (first line), face, hair, body, hands, clothing [, framing, background]."""
    if o.get('prompt'):
        return o['prompt']
    lines = [o.get('who_line') or 'A clean character reference photo of %s.' % o['who'], o.get('framing') or FRAMING.format(**_pron(o))]
    for k in ('face', 'hair', 'body'):
        if o.get(k):
            lines.append('%s: %s' % (k.capitalize(), o[k]))
    lines.append(SKIN_TURBO if o.get('turbo') else SKIN)
    if o.get('hands'):
        lines.append('Hands: %s' % o['hands'])
    clothing = _no_footwear(o['clothing']) if o.get('clothing') else ''
    if clothing:
        lines.append('Clothing: %s' % clothing)
    lines.append(o.get('background') or BACKGROUND)
    return '\n'.join(lines)


def char_from_photo(o):
    if o.get('prompt'):
        return o['prompt']
    p = _pron(o)
    c = _no_footwear(o['clothing']) if o.get('clothing') else ''
    clothing = ('Clothing: %s' % c) if c else 'Same clothing as in image 1.'
    return PHOTO_TO_REF.format(clothing=clothing, **p)


# Outfit change (wardrobe, boss session #13): the master in a new outfit, written as the FL01_ref_v2 edit (face 0.94,
# session #6). The new clothes come from the text (footwear dropped like any master) and / or an outfit picture as
# image 2 (opts outfit_pic) — "the clothes worn in image 2", as the swap boundary edit that keeps the outfit (session #12).
OUTFIT_FRAME = ("Same studio, light and framing as image 1: waist-up medium shot, the frame ends just below the hips, facing "
                "the camera, arms relaxed, calm neutral expression, natural adult proportions.")
OUTFIT_TEXT = "{Pron} now wears a new outfit instead of the one in image 1: {clothing}."
OUTFIT_PIC = ("{Pron} now wears the clothes worn in image 2 instead of the clothes of image 1{clothing}. Take only the clothes "
              "from image 2, not the person, pose, framing or background of image 2.")


def char_outfit(o, p):
    c = _no_footwear(o.get('clothing', '')).rstrip('.')
    if o.get('outfit_pic'):
        return OUTFIT_FRAME + ' ' + OUTFIT_PIC.format(clothing=(': ' + c) if c else '', **p)
    return OUTFIT_FRAME + ' ' + OUTFIT_TEXT.format(clothing=c, **p)


def char_derive(o):
    if o.get('prompt'):
        return o['prompt']
    p = _pron(o)
    scene = char_outfit(o, p) if o.get('kind') == 'outfit' else \
        DERIVE[o['kind']].format(**p) if o.get('kind') in DERIVE else o.get('scene', '')
    return ' '.join(s for s in (FACE_LOCK.format(**p), scene, o.get('extra', ''), FACE_LOCK_END.format(**p)) if s)


def place_anchor(o):
    if o.get('prompt'):
        return o['prompt']
    return PLACE.format(kind=o.get('kind', 'interior'), space=o['space'], scale=o.get('scale', ''), body=o.get('body', '')).strip()


def place_derive(o):
    if o.get('prompt'):
        return o['prompt']
    return PLACE_DERIVE.format(view=o['view'])


# Objects (기능 10 — robots, vehicles that come back in many scenes): a product reference picture from the card, then
# one picture per direction from it (앞 · 옆 · 뒤 · 3/4), like a place's views. Put into scenes by an edit (image 1 the
# object) or as a 'thing' on a plate / a cut box.
PROP = ("A clean product reference photo of {desc}. The whole {what} in view, centred, front three-quarter view. Plain light "
        "grey seamless studio backdrop, soft even light, realistic, no text, no logos.")
PROP_DERIVE = ("The same {what} as image 1: keep exactly its shape, proportions, materials, colours, markings and every part. "
               "{view} The whole {what} in view, centred, on the same plain light grey studio backdrop with the same soft even "
               "light, no text.")


def prop_ref(o):
    if o.get('prompt'):
        return o['prompt']
    return PROP.format(desc=o['desc'].rstrip('.'), what=o.get('what') or 'object')


def prop_derive(o):
    if o.get('prompt'):
        return o['prompt']
    return PROP_DERIVE.format(view=o['view'].rstrip('.') + '.', what=o.get('what') or 'object')


# Scene image (분경도, production ch. 4): the shot's character image (image 1) + the place (image 2) → the shot's first
# frame, 16:9 at the video size. Drama test 01 (session #7, tests/drama/01) tried the ch. 4 rules (written for
# Qwen-Image-Edit-2509, never tried on 2.1 — boss) on 2.1:
#   place as image 1 + "keep the room" → the room kept exactly but the framing is the place's: full body in a wide room,
#     the medium / close-up camera sentence ignored; the place cropped to the closer view → closer, but the room redrawn;
#   character first, the image that already has the shot size (master = medium, a sheet head panel = close-up), "keep
#     the framing of image 1", the room as image 2 → the shot size, the room and the face (0.68-0.90) all held.
# Name only the part of the outfit that is in the frame: shoes named in a waist-up shot came out as straps at the knees
# or in the hands. The shallow-focus lens sentence (opt blur) blurred the room in close-ups; in medium shots it stayed
# mostly sharp. Session #7 default = no forced blur (boss: blur is not wanted as a look); session #18 (boss, design §2):
# short-drama CU·MCU blurred by default, as all four reference dramas do — the callers (Mac script, scene screen) set
# blur for close shots, medium and wider stay in natural focus. One person per image; several people in one shot are
# not tried yet.
SCENE_LENS = {
    'wide': "Camera: eye level, 24mm lens at f/8, everything from the foreground to the back wall in sharp focus.",
    'medium': "Camera: eye level, 50mm lens at f/5.6: {subj} and the room behind both in natural focus.",
    'close': "Camera: eye level, 85mm lens at f/5.6: the eyes in sharp focus, the room behind still clearly readable.",
}
SCENE_LENS_BLUR = {
    'medium': ("Camera: eye level, 50mm lens at f/2: {subj} in sharp focus, the room behind softly blurred by a shallow "
               "depth of field but still recognisable."),
    'close': ("Camera: eye level, 85mm lens at f/1.8: the eyes in sharp focus, the room behind strongly blurred into soft "
              "shapes and colours by a very shallow depth of field."),
}


def scene_compose(o):
    """scene.compose: image_1 = the character image of the shot size, image_2 = the place. o: gender, frame (what the
    crop of image 1 shows, e.g. 'from the top of her head down to the upper thigh'), action (pose, where, expression —
    small), behind (what of the room is behind the person), clothing (only the visible part), shot wide|medium|close,
    blur (shallow depth of field, off by default), light, extra. The face check compares with image_1."""
    if o.get('prompt'):
        return o['prompt']
    p = _pron(o)
    obj = 'her' if p['pron'] == 'she' else 'him'
    lines = ['Keep the %s from image 1 as exactly the same person: same face, eye shape and double eyelids, eyebrows, '
             'nose, lips, jawline, same hair colour, length and style, same skin tone and body shape.' % p['who'],
             'Keep the framing of image 1: the same camera distance and crop, %s.' % o['frame'].rstrip('.'),
             o['action'],
             'Replace the plain grey studio backdrop with the room from image 2, so that %s is inside that room: behind %s '
             'are %s.' % (p['pron'], obj, o['behind'].rstrip('.'))]
    if o.get('clothing'):
        lines.append('Clothing, only what is in the frame: %s.' % o['clothing'].rstrip('.'))
    shot = o.get('shot', 'medium')
    lens = SCENE_LENS_BLUR if o.get('blur') and shot in SCENE_LENS_BLUR else SCENE_LENS
    lines.append(lens[shot].format(subj=obj))
    if o.get('light'):
        lines.append('Light %s like the room: %s.' % (obj, o['light'].rstrip('.')))
    if o.get('extra'):
        lines.append(o['extra'])
    lines.append('%s face is sharp with natural skin texture and fine pores. %s must be instantly recognizable as the %s '
                 'in image 1.' % (p['pos'].capitalize(), p['Pron'], p['who']))
    return '\n'.join(lines)


# Scene image on a fixed background plate (drama test 02): image 1 = the plate (one wide view of the set, cut from the
# place anchor), image 2.. = the people (o['people'], one dict each). The plate is image 1 so the room stays as it is
# (test 01: place as image 1 kept the room exactly but kept its camera too); written as a change to image 1 (Qwen edit
# rewrite prompt, docs/drama.md §5). On a tight plate (one sofa seat filling the frame) the people came out at the wrong
# scale (sitting on the sofa back), so people are seated on a wide plate and the shots are zoomed out of it (scene.zoom).
def _people(o):
    ps = o.get('people') or [{k: o.get(k) for k in ('gender', 'child', 'who', 'frame', 'action', 'clothing')}]
    return [(_pron(q), q.get('frame'), q['action'], q.get('clothing'), k + 2) for k, q in enumerate(ps)]


def scene_plate(o):
    """scene.plate: o: plate (what image 1 shows), people [{gender, frame (how much of the person is seen), action,
    clothing}] for image 2, 3, …, extra. The face check compares with image_2."""
    if o.get('prompt'):
        return o['prompt']
    people = _people(o)
    lines = ['Edit image 1, %s: keep it exactly as it is — the same camera position, angle, framing and lens, '
             'and every object, colour and light in it.' % o['plate'].rstrip('.')]
    for p, frame, action, clothing, k in people:
        lines.append('Add the %s from image %d into image 1: %s %s is seen %s, at the size the camera distance of '
                     'image 1 gives.' % (p['who'], k, action.rstrip('.') + '.', p['Pron'], frame.rstrip('.')))
        lines.append('Keep the %s from image %d as exactly the same person: same face, eye shape and double eyelids, '
                     'eyebrows, nose, lips, jawline, same hair colour, length and style, same skin tone and body shape.'
                     % (p['who'], k))
        if clothing:
            lines.append('%s clothing, only what is in the frame: %s.' % (p['pos'].capitalize(), clothing.rstrip('.')))
    for k, th in enumerate(o.get('things') or []):   # objects (robots, vehicles) after the people's images
        n = len(people) + 2 + k
        lines.append('Add the %s from image %d into image 1: %s Keep it exactly as in image %d: the same shape, proportions, '
                     'colours, materials and every part, at the size the camera distance of image 1 gives.'
                     % (th.get('what') or 'object', n, th['action'].rstrip('.') + '.', n))
    lines.append('Match the light direction, colour, perspective and scale of image 1, with soft contact shadows. '
                 'Natural focus: the person and the background behind both clearly readable.')
    if o.get('extra'):
        lines.append(o['extra'])
    for p, _, _, _, k in people:
        lines.append('%s face is sharp with natural skin texture and fine pores. %s must be instantly recognizable as '
                     'the %s in image %d.' % (p['pos'].capitalize(), p['Pron'], p['who'], k))
    return '\n'.join(lines)


# A shot zoomed out of the seated wide scene (drama test 02): image 1 = a crop of it (soft, the people small), image 2.. =
# each person's reference (master or an expression panel). Same set, same seats and scale in every shot of the scene.
def scene_zoom(o):
    """scene.zoom: o: what (image 1's content), people [{gender, where (e.g. 'the woman on the left'), action (what
    changes, optional), clothing}] for image 2, 3, …, frame (the shot size), blur (CU·MCU: the background behind softly
    out of focus). The face check compares with image_2."""
    if o.get('prompt'):
        return o['prompt']
    lines = ['Image 1 is a small, soft crop of a photo of %s. Re-render image 1 as a sharp, detailed, high-resolution '
             'photo of the same view: keep the framing (%s), the camera angle and position, the set, every object, the '
             'colours and the light, and each person\'s place, size and posture.' % (o['what'].rstrip('.'),
                                                                                     o['frame'].rstrip('.'))]
    for k, q in enumerate(o['people']):
        p, n = _pron(q), k + 2
        lines.append('%s in image 1 is the %s from image %d: give %s exactly the face, eye shape and double eyelids, '
                     'eyebrows, nose, lips, jawline, hair colour, length and style, skin tone and body shape of the %s in '
                     'image %d.' % (q['where'][0].upper() + q['where'][1:], p['who'], n,
                                    'her' if p['pron'] == 'she' else 'him', p['who'], n))
        if q.get('action'):
            lines.append(q['action'].rstrip('.') + '.')
        # a shot above the feet drops the footwear the plate's full-length clothing names: with the shoes in the text a
        # head-to-hips crop came out full length, face 0.39 -> 0.50 without them (session #14, drama test of scene 10)
        clothing = q.get('clothing') if FULL_FRAME.search(o['frame']) else _no_footwear(q.get('clothing') or '')
        if clothing:
            lines.append('%s clothing, only what is in the frame: %s.' % (p['pos'].capitalize(), clothing.rstrip('.')))
    for k, th in enumerate(o.get('things') or []):
        lines.append('The %s in image 1 is the one from image %d: keep exactly its shape, proportions, colours, materials '
                     'and every part.' % (th.get('what') or 'object', len(o['people']) + 2 + k))
    lines.append('Shallow depth of field like an 85mm lens at f/1.8: the faces in sharp focus, the background behind them '
                 'strongly blurred into soft shapes and colours.' if o.get('blur') else
                 'Natural focus: the people and the background behind them clearly readable.')
    for k, q in enumerate(o['people']):
        p = _pron(q)
        lines.append('%s face is sharp with natural skin texture and fine pores. %s must be instantly recognizable as '
                     'the %s in image %d.' % (p['pos'].capitalize(), p['Pron'], p['who'], k + 2))
    return '\n'.join(lines)


def sheet_panels(o):
    """char.sheet: the 8 panels of an H3 reference sheet derived from the master (image 1) → [(kind, prompt, retries)].
    Heads 2x2 (front, 3/4 left, 3/4 right, profile) + full bodies (front, 3/4, profile, back) in the card's outfit.
    Retries only where the face check is meaningful (front-ish faces); profile and back panels are drawn once."""
    p = _pron(dict(gender=o.get('gender', 'f')))
    P, pos, who = p['Pron'], p['pos'], p['who']
    lock, end = FACE_LOCK.format(**p), FACE_LOCK_END.format(**p)
    light = ('Keep the %s from image 1 as exactly the same person: same face, same hair colour, length and cut, same skin '
             'tone, same body shape.' % who)
    outfit = 'Outfit, exactly as in image 1 and complete: %s' % o.get('clothing', '')
    full = ('Same studio and light. Full-body shot from the top of the head to the soles of the shoes, standing upright, '
            'arms relaxed, the whole body visible with space above the head and below the feet, natural adult proportions '
            'about eight heads tall. ')
    head = 'Same studio, light and outfit. Head-and-shoulders close-up, calm neutral expression. '
    turn = ('%s turns %s head and shoulders about 45 degrees to %s own %%s, towards the %%s side of the image: three-quarter '
            'view, the far cheek partly hidden, the nose crossing the far cheek line, eyes looking at the lens.' % (P, pos, pos))
    return [
        ('sh_head_front', ' '.join([lock, head + 'Front view, looking into the lens.', end]), 3),
        ('sh_head_34L', ' '.join([lock, head + turn % ('left', 'right'), end]), 3),
        ('sh_head_34R', ' '.join([lock, head + turn % ('right', 'left'), end]), 3),
        ('sh_head_side', ' '.join([light, head + 'Strict 90-degree profile: %s face points to the right side of the image, '
                                   'only one eye and one ear visible, the profile line of forehead, nose, lips and chin sharp '
                                   'against the backdrop.' % pos]), 0),
        ('sh_full_front', ' '.join([lock, full + 'Front view, facing the camera.', outfit, end]), 3),
        ('sh_full_34', ' '.join([light, full + 'Three-quarter view: %s body and face turned about 45 degrees to %s own left, '
                                 'towards the right side of the image.' % (pos, pos), outfit]), 0),
        ('sh_full_side', ' '.join([light, full + 'Strict 90-degree left profile: %s faces the right side of the image, only '
                                   'the left side of the face and body visible.' % p['pron'], outfit]), 0),
        ('sh_full_back', ' '.join([light, full + 'Seen strictly from behind: only the back of the head, the back and the '
                                   'heels visible, same hair seen from the back.', outfit]), 0),
    ]


# Expressions (production ch. 2: 8 expressions + an open speaking mouth), each in three strengths: 0 subtle, 1 medium
# (default, boss session #10), 2 strong. Narrow, concrete face cues — production: 'tense' alone gave wide eyes and an open
# mouth. Whether 2.1 keeps the face through an expression edit is still to be checked (design §11).
EXPR = {
    'calm': ('a calm, composed expression, face relaxed, mouth closed',
             'a calm, composed expression, steady gaze, mouth closed',
             'a deeply calm, serene expression, soft steady gaze, mouth closed'),
    'curious': ('a faint hint of curiosity: eyebrows lifted very slightly, eyes a little more open, mouth closed',
                'a curious look: eyebrows slightly raised, attentive eyes, head tilted a little, mouth closed',
                'open curiosity: eyebrows raised, eyes wide with interest, head tilted, lips slightly parted'),
    'tense': ('slight tension: lips pressed together lightly, a small crease between the brows',
              'a tense look: jaw tight, lips pressed together, brows drawn together, eyes alert',
              'strong tension: jaw clenched, lips pressed thin, brows drawn hard together, eyes narrowed'),
    'surprise': ('mild surprise: eyebrows lifted slightly, eyes a little wider, lips just parted',
                 'surprise: eyebrows raised, eyes wide open, lips parted',
                 'strong surprise: eyebrows raised high, eyes wide, mouth open'),
    'fear': ('a trace of unease: inner brows raised slightly, eyes a little wide, lips pressed',
             'fear: brows raised and drawn together, eyes wide, lips slightly parted and tense',
             'strong fear: brows raised and pinched together, eyes wide, mouth open, tense neck'),
    'sad': ('a hint of sadness: inner brows raised slightly, eyes a little downcast, mouth corners turned down slightly',
            'sadness: inner brows raised, eyes downcast and glossy, mouth corners turned down',
            'deep sadness: brows pinched upward, eyes wet and downcast, lips turned down'),
    'resolve': ('quiet resolve: steady direct gaze, lips closed firmly',
                'determination: steady gaze, brows level and slightly lowered, lips firmly closed, jaw set',
                'fierce determination: intense gaze, brows lowered, lips pressed firmly, jaw clenched'),
    'relax': ('relaxed: soft eyes, loose face, mouth closed',
              'relaxed and at ease: soft eyes, loose face muscles, a faint gentle smile',
              'completely at ease: soft half-lidded eyes, loose face, a warm gentle smile'),
    'talk': ('mouth slightly open as if speaking mid-word, neutral expression',) * 3,
}


def expr_prompts(o):
    """char.sheet option expr [ids] + strength 0/1/2 → [(kind ex_<id>, prompt, retries)]: head-and-shoulders close-ups
    from the master (image 1), saved as separate images next to the sheet. expr_custom [{id, text}] (기능 12): more
    panels written as a sentence of their own (English id, e.g. smirk → ex_smirk), the same frame and face lock."""
    p = _pron(dict(gender=o.get('gender', 'f'), child=o.get('child')))
    s = min(2, max(0, int(o.get('strength', 1))))
    head = 'Same studio, light and outfit. Head-and-shoulders close-up, front view, looking into the lens. Expression: '
    out = [(k, EXPR[k][s]) for k in (o.get('expr') or []) if k in EXPR]
    out += [(str(x['id']), str(x['text']).strip().rstrip('.')) for x in o.get('expr_custom') or []
            if re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)*', str(x.get('id', ''))) and str(x.get('text', '')).strip()
            and str(x['id']) not in EXPR]
    return [('ex_' + k, ' '.join([FACE_LOCK.format(**p), head + text + '.', FACE_LOCK_END.format(**p)]), 2) for k, text in out]

# Face check thresholds (#18). Measured on FL01 alone (2026-11-20): same person, frontal views 0.80~0.91, 3/4 angle
# 0.60~0.72, other people 0.27~0.56 (a frontal view of a similar age is 0.56). Re-measure as characters are added.
MIN_SIM = {'angle_34L': 0.5, 'angle_34R': 0.5, 'side_L': 0.35, 'side_R': 0.35}
MIN_SIM_FRONT = 0.65
MIN_SIM_PHOTO = 0.6


def min_sim(kind, o):
    if kind == 'char.from_photo':
        return MIN_SIM_PHOTO
    return MIN_SIM.get(o.get('kind'), MIN_SIM_FRONT)


# job type → (prompt function, default settings). Sizes per production rules: master 4MP 2:3, place 16:9.
JOBS = {
    'char.ref':        (char_ref,        {'aspect': '2:3', 'mp': 4.0}),
    'char.from_photo': (char_from_photo, {'aspect': '2:3', 'mp': 4.0, 'check': 'image_1'}),
    'char.derive':     (char_derive,     {'check': 'image_1'}),
    'place.anchor':    (place_anchor,    {'aspect': '16:9', 'mp': 1.0}),
    'place.derive':    (place_derive,    {'aspect': '16:9', 'mp': 1.0}),
    'prop.ref':        (prop_ref,        {'aspect': '2:3', 'mp': 1.0}),
    'prop.derive':     (prop_derive,     {'aspect': '2:3', 'mp': 1.0}),
    # scene image at the video size (16:9 as the place, production ch. 4; LTX 1920x1088, H3 crops to its canvas)
    'scene.compose':   (scene_compose,   {'width': 1920, 'height': 1088, 'check': 'image_1'}),
    'scene.plate':     (scene_plate,     {'width': 1920, 'height': 1088, 'check': 'image_2'}),
    'scene.zoom':      (scene_zoom,      {'width': 1920, 'height': 1088, 'check': 'image_2'}),
}


# ---- person swap (video.swap, #8) ----
# boundary frame edit: image 1 = source boundary frame, image 2 = character master (seam ①: segment start/end frames).
# Identity from image 2, pose from image 1 — "the body must match image 2" made the edit copy the master's standing pose
# (FL02 29 s swap, session #7: three boundaries redrawn arms-down, H3 then froze before them)
SWAP_EDIT = ("Replace the person in image 1 with the {who} from image 2, in exactly the pose of image 1. From image 2 "
             "take only who {pron} is: the face (eye shape, eyebrows, nose, lips, jawline, face shape), hair colour and "
             "length, body shape and clothing. From image 1 keep the pose exactly: where each arm, hand, leg and foot is, "
             "the head angle, the body turn, and the position and size in the frame. Keep the camera angle, background "
             "and lighting of image 1. {clothing} {Pron} must be instantly recognizable as the {who} in image 2.")
# boundary frames without the feet (medium shots, close-ups — session #12): image 2 is the master, the clothes are taken
# from it without naming them, the framing spelled out. With SWAP_EDIT and the full-body panel, Qwen redrew a medium
# two-shot and a 9:16 close-up as zoomed-out full-body pictures (swap 16·17); the card clothing sentence alone (it names
# trousers and shoes) did the same even with the master; without any clothes words the framing held but the source
# clothes stayed. This wording kept the framing (face size 1.09 / 0.79 of the source) and put FL02's blazer on
# (session #12 edit-only test, the sentence "Do not show more …" got the close-up through one draw earlier)
SWAP_EDIT_PART = ("Replace the person in image 1 with the {who} from image 2, dressed in the clothes {pron} wears in "
                  "image 2 instead of the clothes of image 1. From image 2 take who {pron} is: the face (eye shape, "
                  "eyebrows, nose, lips, jawline, face shape), hair colour and length, and the clothes. Keep everything "
                  "else of image 1 exactly: the same framing and shot size - the edges of the picture cut {obj} at the "
                  "same places as in image 1 and {pos} head is the same size and in the same place - and the same pose, "
                  "head angle, facial expression and mouth shape, camera angle, background and lighting. Do not show "
                  "more of {pos} body than image 1 shows. {Pron} must be instantly recognizable as the {who} in image 2.")
# new place for the swap (#10). Text only (opts place): the person edit also redraws the background (SWAP_EDIT_PLACE).
# Background image (inputs background): two edits per boundary — the person edit (SWAP_EDIT, keeps the pose), then
# SWAP_BG puts the same image behind her on every boundary. One edit with three images ignored the pose and drew the
# master's standing pose (or ignored image 3 with the pose-first wording) — session #7 test
SWAP_EDIT_PLACE = ("Replace the person in image 1 with the {who} from image 2, in exactly the pose of image 1. From image 2 "
                   "take only who {pron} is: the face (eye shape, eyebrows, nose, lips, jawline, face shape), hair colour "
                   "and length, body shape and clothing. From image 1 keep the pose exactly: where each arm, hand, leg and "
                   "foot is, the head angle, the body turn, the position and size in the frame, and the camera angle. "
                   "Replace the whole background with {place}; light {obj} with its lighting. {clothing} {Pron} must be "
                   "instantly recognizable as the {who} in image 2.")
SWAP_BG = ("Replace the whole background of image 1 with the background of image 2{place}. Everything behind the {who} "
           "- the ground, buildings, walls, trees, sky and lights - comes from image 2, and nothing of the background of "
           "image 1 remains. Keep the {who} in image 1 exactly as {pron} is: the same pose (every arm, hand, leg and "
           "foot), head angle, face, hair, body, clothing, and the same position and size in the frame. Keep the camera "
           "angle of image 1. Light {obj} with the lighting of image 2.")
# H3 Ref2VA, official 6-section format. <Video 1> partially_preserved + original face/clothes not kept (checked in
# production ch. 5)
SWAP_H3 = """subject_definitions:
<Subject 1> is the {who} in <Picture 1>{desc}.
<Video 1> is the source footage.

summary:
[video editing + reference generation] The target video is an edited version of <Video 1> in which the person in <Video 1> is replaced by <Subject 1>.

retention_analysis:
<Video 1>: partially_preserved - the motion, poses, gestures, timing, lip movements, camera movement, framing, background and lighting are kept; the original person's face, hair and clothes are not kept.
<Subject 1>: fully_preserved - {pos} face, hair, build and clothing match <Picture 1>.

detailed_description:
[Shot 1] {scene}

overall_soundscape:
The same sound as <Video 1>.

non_diegetic_music:
None."""


# Two people (#11, session #7): no boundary edits (they redrew close-ups as full-body shots), so only the H3 prompt.
# The people to replace are named by their source look (opts person_a / person_b: the dancers wear the same outfit,
# hair tells them apart). One may be drawn taller (opts taller = 'b', boss: the man is taller than the woman)
SWAP2_H3 ="""subject_definitions:
<Subject 1> is the {who_a} in <Picture 1>{desc_a}.
<Subject 2> is the {who_b} in <Picture 2>{desc_b}.
<Video 1> is the source footage.

summary:
[video editing + reference generation] The target video is an edited version of <Video 1> in which {person_a} is replaced by <Subject 1> and {person_b} is replaced by <Subject 2>.{taller}

retention_analysis:
<Video 1>: partially_preserved - the motion, poses, gestures, timing, lip movements, camera movement, framing, background and lighting are kept; the original people's faces, hair and clothes are not kept{logo}.
<Subject 1>: fully_preserved - {pos_a} face, hair, build and clothing match <Picture 1>.
<Subject 2>: fully_preserved - {pos_b} face, hair, build and clothing match <Picture 2>.

detailed_description:
[Shot 1] The same shot as <Video 1>, performed by <Subject 1> and <Subject 2>, matching every movement and expression of <Video 1>: <Subject 1> moves exactly as {person_a} does and <Subject 2> exactly as {person_b} does.{taller}

overall_soundscape:
The same sound as <Video 1>.

non_diegetic_music:
None."""


def swap2_prompt(o):
    """Two-person swap: the H3 prompt. o: gender/desc for A, gender_b/desc_b for B, person_a / person_b (who they are
    in the source), taller ('b' or 'a'), clean_logo."""
    pa = _pron(o)
    pb = _pron({'gender': o.get('gender_b', 'm'), 'who': o.get('who_b')})
    src_a, src_b = o['person_a'].strip().rstrip('.'), o['person_b'].strip().rstrip('.')
    tall = str(o.get('taller') or '').lower()
    taller = {'b': ' <Subject 2> is clearly taller than <Subject 1>, about half a head taller.',
              'a': ' <Subject 1> is clearly taller than <Subject 2>, about half a head taller.'}.get(tall, '')
    d = lambda k: (': ' + o[k].rstrip('.')) if o.get(k) else ''
    h3 = SWAP2_H3.format(who_a=pa['who'], who_b=pb['who'], desc_a=d('desc'), desc_b=d('desc_b'), person_a=src_a,
                         person_b=src_b, pos_a=pa['pos'], pos_b=pb['pos'], taller=taller,
                         logo=' and the watermark of <Video 1> is not kept' if o.get('clean_logo') else '')
    return o.get('prompt') or h3


def swap_prompts(o, bg_image=False):
    """(person edit prompt, person edit prompt for boundary frames without the feet (SWAP_EDIT_PART — there image 2 is
    the master or a head panel, worker), H3 prompt, background edit prompt or None). bg_image: a background image
    follows the person edit on every boundary (SWAP_BG); o['place'] (text) describes the new place (alone: the person
    edit redraws it)."""
    p = _pron(o)
    obj = 'her' if p['pron'] == 'she' else 'him'
    clothing = 'Clothing: %s.' % o['clothing'].rstrip('.') if o.get('clothing') else 'The clothing comes from image 2.'
    place = str(o.get('place') or '').strip().rstrip('.')
    if place and not bg_image:
        edit = SWAP_EDIT_PLACE.format(clothing=clothing, place=place, obj=obj, **p)
    else:
        edit = SWAP_EDIT.format(clothing=clothing, **p)
    part = SWAP_EDIT_PART.format(obj=obj, **p)
    if place and not bg_image:
        part = part.replace('camera angle, background and lighting.', 'camera angle. Replace the whole background with '
                            '%s; light %s with its lighting.' % (place, obj), 1)
    bg = SWAP_BG.format(place=' (%s)' % place if place else '', obj=obj, **p) if bg_image else None
    desc = (': ' + o['desc'].rstrip('.')) if o.get('desc') else ''
    env = place or ('the new place' if bg_image else '')
    scene = o.get('scene') or ('The same shot as <Video 1>, performed by <Subject 1>%s, matching every movement, '
                               'expression and lip movement of <Video 1>.' % (' in <Environment>' if env else ''))
    h3 = SWAP_H3.format(desc=desc, scene=scene, **p)
    who_src = str(o.get('person') or '').strip().rstrip('.')   # several people in the source: which one is replaced
    if who_src:   # (UI 사람 표, session #10 — swap 17)
        edit, part = (e.replace('Replace the person in image 1', 'Replace %s in image 1 (keep everyone else as they are)'
                                % who_src, 1) for e in (edit, part))
        h3 = h3.replace('in which the person in <Video 1> is replaced by <Subject 1>',
                        'in which %s in <Video 1> is replaced by <Subject 1>; everyone else in <Video 1> stays as they are' % who_src, 1)
    if env:   # new place: the original background is not kept; the place comes from the edited start/end frames
        h3 = h3.replace('<Video 1> is the source footage.',
                        '<Video 1> is the source footage.\n<Environment>: %s, as in the first and last frames.' % env)
        h3 = h3.replace('is replaced by <Subject 1>.', 'is replaced by <Subject 1> and the scene is moved to <Environment>.')
        h3 = h3.replace('camera movement, framing, background and lighting are kept; the original person\'s face, hair '
                        'and clothes are not kept.', 'camera movement and framing are kept; the original person\'s face, '
                        'hair and clothes and the original background are not kept.')
        h3 = h3.replace('match <Picture 1>.', 'match <Picture 1>.\n<Environment>: newly_generated.')
    return (o.get('edit_prompt') or edit, o.get('edit_prompt_part') or o.get('edit_prompt') or part,
            o.get('prompt') or h3, o.get('bg_prompt') or bg)
