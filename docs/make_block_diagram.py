"""Generate docs/sm_cnn_block_diagram.svg (block design of the fixed SM-CNN).

    python docs/make_block_diagram.py
"""
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else __file__.replace("make_block_diagram.py", "sm_cnn_block_diagram.svg")
W, H = 1640, 1080
P = []

INK, MUTED = '#0f172a', '#475569'
MOD, SKIP, RES = '#c2410c', '#0f766e', '#6d28d9'


def rect(x, y, w, h, fill, stroke='#334155', rx=6, sw=1.2):
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size=13, anchor='middle', weight='normal', fill=INK, italic=False):
    st = ' font-style="italic"' if italic else ''
    P.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{st}>{s}</text>')


def lines(cx, cy, rows, size=13, gap=16, fill=INK, weight='normal'):
    """Centered multi-line label; rows = list of (text, size, fill)."""
    y0 = cy - gap * (len(rows) - 1) / 2 + size * 0.35
    for i, r in enumerate(rows):
        s, sz, f = (r, size, fill) if isinstance(r, str) else r
        text(cx, y0 + i * gap, s, sz, fill=f, weight=weight)


def vtext(cx, cy, s, size=12):
    P.append(f'<text transform="translate({cx},{cy}) rotate(-90)" font-size="{size}" text-anchor="middle" '
             f'dominant-baseline="central" fill="{INK}">{s}</text>')


def path(d, color='#334155', marker='arr', dash=None, sw=1.6):
    m = f' marker-end="url(#{marker})"' if marker else ''
    ds = f' stroke-dasharray="{dash}"' if dash else ''
    P.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{ds}{m}/>')


def circle(cx, cy, r, label, size=14):
    P.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="#334155" stroke-width="1.4"/>')
    text(cx, cy + size * 0.36, label, size, weight='bold')


def sup(base, s):
    return f'{base}<tspan dy="-6" font-size="75%">{s}</tspan><tspan dy="6">​</tspan>'


YL = sup('y', 'λ')
YS = sup('y', 's')
XS = sup('x̂', 's')

# ---------------------------------------------------------------- header
text(32, 36, 'SM-CNN denoiser — fixed implementation (K = 24, 2,404,361 trainable parameters)', 20, 'start', 'bold')
text(32, 60, 'Torun et al., Signal Processing 214 (2024) 109248 · code: model/model.py · every block below has its own weights',
     13, 'start', fill=MUTED)

# ---------------------------------------------------------------- (a) overall network
rect(16, 76, 1608, 470, '#f8fafc', '#cbd5e1', rx=10)
text(32, 100, '(a) Overall network', 15, 'start', 'bold')

# legend
lx, ly = 1060, 100
for i, (label, color, dash) in enumerate([('feature flow', '#334155', None), (f'modulation ({YL})', MOD, None),
                                          ('skip connection', SKIP, '5 4'), ('global residual', RES, '7 4')]):
    x = lx + i * 140
    path(f'M{x},{ly - 4} H{x + 30}', color, {'#334155': 'arr', MOD: 'arrMod', SKIP: 'arrSkip', RES: 'arrRes'}[color], dash)
    text(x + 38, ly, label, 12, 'start', fill=MUTED)

# inputs
rect(40, 150, 136, 84, '#fdba74', '#c2410c')
lines(108, 192, [('K adjacent noisy bands', 12, INK), (f'{YL}   1 × K × h × w', 13, INK),
                 ('target band at index K//2', 11, MUTED)], gap=18)
rect(40, 340, 136, 64, '#fde68a', '#b45309')
lines(108, 372, [('noisy band to denoise', 12, INK), (f'{YS}   1 × h × w', 13, INK)], gap=18)

# spectral branch
for i, k in enumerate((3, 5, 7)):
    cy = 164 + 34 * i
    rect(206, cy - 14, 132, 28, '#fed7aa', '#c2410c')
    text(272, cy + 4.5, f'conv3d K×{k}×{k} → 20', 12)
    path(f'M176,192 L206,{cy}')
    path(f'M338,{cy} H362')
rect(362, 146, 66, 116, '#e2e8f0')
lines(395, 204, ['concat', '+ ReLU', ('60', 12, MUTED)], 12, gap=17)

# spatial branch
for i, k in enumerate((3, 5, 7)):
    cy = 344 + 34 * i
    rect(206, cy - 14, 132, 28, '#fef3c7', '#b45309')
    text(272, cy + 4.5, f'conv2d {k}×{k} → 20', 12)
    path(f'M176,372 L206,{cy}')
    path(f'M338,{cy} H362')
rect(362, 326, 66, 116, '#e2e8f0')
lines(395, 384, ['concat', '+ ReLU', ('60', 12, MUTED)], 12, gap=17)

# merge + head
circle(490, 294, 16, 'C')
text(518, 287, '120', 11, fill=MUTED)
path('M428,204 H490 V276')
path('M428,384 H490 V312')
rect(530, 244, 50, 100, '#bbf7d0', '#15803d')
vtext(555, 294, 'conv 3×3 + ReLU', 12)
text(555, 360, 'head', 11, fill=MUTED)
path('M506,294 H530')

# stages
X0 = [610 + 186 * s for s in range(4)]
path(f'M580,294 H{X0[0]}')
path(f'M108,150 V124 H{X0[3] + 83}', MOD, None)
text(820, 114, f'{YL} (squeezed to K × h × w) modulates every SSMRB → 16 SSMMs, each with its own convolutions',
     12, fill=MOD)
for s, x0 in enumerate(X0):
    for j, bx in enumerate((x0, x0 + 60)):
        rect(bx, 244, 46, 100, '#bfdbfe', '#1d4ed8')
        vtext(bx + 23, 294, 'SSMRB', 13)
        path(f'M{bx + 23},124 V244', MOD, 'arrMod')
    rect(x0 + 120, 244, 46, 100, '#bbf7d0', '#15803d')
    vtext(x0 + 143, 294, 'conv 3×3 + ReLU', 12)
    path(f'M{x0 + 46},294 H{x0 + 60}')
    path(f'M{x0 + 106},294 H{x0 + 120}')
    if s < 3:
        path(f'M{x0 + 166},294 H{X0[s + 1]}')
    text(x0 + 53, 362, f'stage {s + 1}', 12, fill=MUTED)
    # skip conv
    rect(x0 + 113, 384, 60, 44, '#99f6e4', SKIP)
    lines(x0 + 143, 406, [('conv 3×3', 11, INK), ('→ 15', 11, INK)], gap=14)
    path(f'M{x0 + 143},344 V384', SKIP, 'arrSkip', '5 4')
    path(f'M{x0 + 143},428 V456', SKIP, None, '5 4')
path(f'M{X0[0] + 143},456 H1350', SKIP, 'arrSkip', '5 4')
text(X0[0] + 150, 476, 'one skip per stage, each through its own conv 3×3 (60 → 15)', 12, 'start', fill=SKIP)

# tail + output
circle(1366, 456, 16, 'C')
text(1366, 432, '60', 11, fill=MUTED)
rect(1398, 431, 116, 50, '#e2e8f0')
lines(1456, 456, ['ReLU → conv 3×3', ('60 → 1', 12, MUTED)], 12, gap=17)
path('M1382,456 H1398')
circle(1564, 456, 14, '+', 18)
path('M1514,456 H1550')
rect(1512, 260, 104, 68, '#c7d2fe', '#4338ca')
lines(1564, 294, [('denoised band', 12, INK), (f'{XS}   1 × h × w', 13, INK)], gap=18)
path('M1564,442 V328')
path('M108,404 V512 H1564 V470', RES, 'arrRes', '7 4')
text(836, 530, f'global residual learning: {XS} = network output + {YS}', 12, fill=RES)

# ---------------------------------------------------------------- (b) SSMRB
rect(16, 562, 800, 300, '#f8fafc', '#cbd5e1', rx=10)
text(32, 586, '(b) Spectral self-modulating residual block (SSMRB)', 15, 'start', 'bold')
text(70, 690, 'x', 13, italic=True)
text(70, 724, '60×h×w', 11, fill=MUTED)
path('M40,700 H130')
path('M110,700 V632 H744 V686')
text(430, 624, 'identity skip', 12, fill=MUTED)
blocks = [(130, 'SSMM₁', '#fbcfe8', '#be185d', None), (236, 'LeakyReLU', '#fef9c3', '#a16207', 'slope 0.02'),
          (330, 'conv 3×3', '#dcfce7', '#15803d', '60 → 60'), (436, 'SSMM₂', '#fbcfe8', '#be185d', None),
          (542, 'LeakyReLU', '#fef9c3', '#a16207', 'slope 0.02'), (636, 'conv 3×3', '#dcfce7', '#15803d', '60 → 60')]
widths = {'SSMM₁': 84, 'SSMM₂': 84, 'LeakyReLU': 72, 'conv 3×3': 84}
for i, (x, label, fill, stroke, sub) in enumerate(blocks):
    w = widths[label]
    rect(x, 672, w, 56, fill, stroke)
    lines(x + w / 2, 700, [label] + ([(sub, 11, MUTED)] if sub else []), 13 if sub is None else 12, gap=16)
    nxt = blocks[i + 1][0] if i + 1 < len(blocks) else 730
    path(f'M{x + w},700 H{nxt}')
circle(744, 700, 14, '+', 18)
path('M758,700 H806')
text(784, 690, 'out', 12, fill=MUTED)
text(36, 794, f'{YL} (K×h×w)', 13, 'start', fill=MOD)
path('M124,790 H478', MOD, None)
path('M172,790 V728', MOD, 'arrMod')
path('M478,790 V728', MOD, 'arrMod')
text(36, 840, '266,880 parameters per SSMRB · two SSMMs and two convolutions, none shared with other blocks',
     12, 'start', fill=MUTED)

# ---------------------------------------------------------------- (c) SSMM
rect(832, 562, 792, 300, '#f8fafc', '#cbd5e1', rx=10)
text(848, 586, '(c) Spectral self-modulation module (SSMM)', 15, 'start', 'bold')
text(852, 640, sup('f', 'pre'), 13, 'start', italic=True)
path('M850,650 H900')
rect(900, 620, 200, 60, '#fce7f3', '#be185d')
lines(1000, 650, ['instance norm (per channel)', ('(f − μ_c) / √(σ_c² + δ),  δ = 10⁻⁵', 11, MUTED)], 12, gap=18)
circle(1170, 650, 14, '×', 16)
circle(1290, 650, 14, '+', 18)
path('M1100,650 H1156')
path('M1184,650 H1276')
path('M1304,650 H1370')
text(1378, 655, sup('f', 'next'), 13, 'start', italic=True)
text(848, 772, YL, 14, 'start', fill=MOD, italic=True)
text(848, 790, 'K×h×w', 11, 'start', fill=MUTED)
path('M892,776 H930', MOD, 'arrMod')
rect(930, 748, 150, 56, '#fbcfe8', '#be185d')
lines(1005, 776, ['conv 5×5, K → 60', ('+ ReLU (shared trunk)', 11, MUTED)], 12, gap=17)
path('M1080,776 H1245')
path('M1100,776 V720 H1125')
rect(1125, 700, 90, 40, '#fbcfe8', '#be185d')
text(1170, 724, 'conv 3×3 → γ', 12)
rect(1245, 756, 90, 40, '#fbcfe8', '#be185d')
text(1290, 780, 'conv 3×3 → β', 12)
path('M1170,700 V664')
text(1180, 688, '1 + γ', 12, 'start')
path('M1290,756 V664')
text(1300, 712, 'β', 12, 'start')
text(1420, 702, 'γ, β: 60 × h × w', 12, 'start', fill=MUTED)
text(1420, 720, '(per pixel, per channel)', 12, 'start', fill=MUTED)
text(848, 828, f'{sup("f", "next")} = norm({sup("f", "pre")}) ⊙ (1 + γ({YL})) + β({YL})    (Eq. 3, with γ re-parameterised as 1 + γ)',
     12, 'start')
text(848, 848, '100,980 parameters per SSMM · γ and β come from two separate conv 3×3 heads', 12, 'start', fill=MUTED)

# ---------------------------------------------------------------- (d) parameters + fixes
rect(16, 878, 1608, 186, '#f8fafc', '#cbd5e1', rx=10)
text(32, 902, '(d) Parameters (K = 24)', 15, 'start', 'bold')
rows = [('Spatial branch: 3 × conv2d (1 → 20)', '1,720'), ('Spectral branch: 3 × conv3d (1 → 20)', '39,900'),
        ('Head: conv 3×3, 120 → 60', '64,860'), ('8 × SSMRB (266,880 each)', '2,135,040'),
        ('4 × stage conv 3×3, 60 → 60', '129,840'), ('4 × skip conv 3×3, 60 → 15', '32,460'),
        ('Tail: conv 3×3, 60 → 1', '541'), ('Total (= Table 5 of the paper)', '2,404,361')]
for i, (name, n) in enumerate(rows):
    y = 926 + 17 * i
    bold = 'bold' if i == len(rows) - 1 else 'normal'
    text(40, y, name, 12.5, 'start', bold)
    text(410, y, n, 12.5, 'end', bold)
P.append('<line x1="40" y1="1034" x2="410" y2="1034" stroke="#94a3b8" stroke-width="1"/>')
text(460, 902, 'Fixed relative to the previous implementation', 15, 'start', 'bold')
fixes = ['Every SSMRB, SSMM, stage conv and skip conv has its own weights (the old code reused one of each: 248,576 parameters).',
         'γ and β are predicted by two separate conv 3×3 heads (the old code used one convolution, so γ was equal to β).',
         'The noisy input band and the clean target are the same band, index K//2 (old: input band 11, target band 12).',
         'Instance norm uses the biased variance of Eqs. 4–5; the constructor takes keyword arguments (old call swapped the 2D/3D filter counts).',
         'Inference mirror-pads K//2 bands, slides a K-band window and denoises each full band, for any image size or band count.']
for i, f in enumerate(fixes):
    y = 928 + 25 * i
    P.append(f'<circle cx="466" cy="{y - 4}" r="3" fill="#334155"/>')
    text(478, y, f, 12.5, 'start')

marker = lambda mid, color: (f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Segoe UI, Helvetica, Arial, sans-serif">\n'
       f'<title>SM-CNN block diagram (fixed implementation)</title>\n'
       f'<defs>{marker("arr", "#334155")}{marker("arrMod", MOD)}{marker("arrSkip", SKIP)}{marker("arrRes", RES)}</defs>\n'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>\n' + '\n'.join(P) + '\n</svg>\n')
with open(OUT, 'w', encoding='utf-8') as fp:
    fp.write(svg)
print('wrote', OUT)
