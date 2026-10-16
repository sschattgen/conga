"""Tests for conga/tcrdist/mpl_logo_basic.py, the matplotlib renderer for the
logos in the graph-vs-graph / tcr-clumping figures (svg_free branch).

The layout code (make_tcr_logo, make_rank_genes_logo_stack, ...) is shared by
the legacy SVG renderer and the matplotlib renderer via a `drawer` argument.
These tests check:
  * that, for real TCR logos and DEG/TCR-feature logos, the two drawers are
    handed exactly the same boxes, glyphs, rectangles and colors, in the same
    order (parsed back out of the SVG text), and
  * that the matplotlib rendering of each primitive follows the SVG geometry
    conventions (glyph scaling into its box, stroke centered on the edge).
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from conga import plotting, svg_basic
from conga.tcrdist import mpl_logo_basic, tcrdist_svg_basic
from conga.tcrdist.make_tcr_logo import (
    make_tcr_logo_svg_commands_for_tcrs, draw_tcr_logo_for_tcrs,
    xmargin, ymargin, default_width, default_height)
from conga.tcrdist.tcr_distances import TcrDistCalculator

REPO_ROOT = Path(__file__).resolve().parent.parent
CLONES_FILE = REPO_ROOT / 'test_data' / 'classic_clones.tsv'


def _load_tcrs(mask_fxn, max_n):
    df = pd.read_csv(CLONES_FILE, sep='\t')
    df = df[mask_fxn(df)].head(max_n)
    assert df.shape[0] >= 5
    tcrs = []
    for l in df.itertuples():
        for x in [l.va_gene, l.ja_gene, l.cdr3a, l.cdr3a_nucseq,
                  l.vb_gene, l.jb_gene, l.cdr3b, l.cdr3b_nucseq]:
            assert isinstance(x, str) and x
        tcrs.append(((l.va_gene, l.ja_gene, l.cdr3a, l.cdr3a_nucseq),
                     (l.vb_gene, l.jb_gene, l.cdr3b, l.cdr3b_nucseq)))
    return tcrs


TCR_SETS = {
    # similar TCRs (MAIT-like): tall, conserved logo columns
    'trav1-2': lambda: _load_tcrs(lambda df: df.va_gene.str.startswith('TRAV1-2*'), 12),
    # diverse TCRs: mixed CDR3 lengths --> gap columns
    'diverse': lambda: _load_tcrs(lambda df: np.full(df.shape[0], True), 15),
}


@pytest.fixture(scope='module')
def tcrdist_calculator():
    return TcrDistCalculator('human')


## parse the SVG primitives back into boxes ####################################

_TEXT_IN_BOX_RE = re.compile(
    r'<text x="([-\d.]+)" y="([-\d.]+)" font-size="100" font-family="[^"]+" '
    r'fill="([^"]+)" transform="scale\(([-\d.]+),([-\d.]+)\)">([^<]*)</text>')
_TEXT_RE = re.compile(
    r'<text x="([-\d.]+)" y="([-\d.]+)" font-size="([-\d.e]+)" '
    r'font-weight="(\w+)" font-family="([^"]+)" fill="([^"]+)" '
    r'xml:space="preserve">([^<]*)</text>')
_RECT_RE = re.compile(
    r'<rect x="([-\d.e]+)" y="([-\d.e]+)" height="([-\d.e]+)" width="([-\d.e]+)" '
    r'style="fill:([^;]+);stroke:([^;]+);stroke-width:([-\d.e]+)" />')


def _parse_svg(svg_text):
    ''' Returns list of primitives in document order, in the same form as the
    mpl_logo_basic commands
    '''
    prims = []
    pattern = re.compile('|'.join(
        f'(?:{r.pattern})' for r in [_TEXT_IN_BOX_RE, _TEXT_RE, _RECT_RE]))
    for m in pattern.finditer(svg_text):
        s = m.group(0)
        if (mm := _TEXT_IN_BOX_RE.fullmatch(s)):
            x, y, color, sx, sy, text = mm.groups()
            x, y, sx, sy = map(float, (x, y, sx, sy))
            # invert text_in_box: x=x0/sx y=y1/sy sx=w/(60*len) sy=h/75
            x0, y1 = x*sx, y*sy
            w, h = sx*60*len(text), sy*75
            prims.append(('box', (x0, y1-h, x0+w, y1), text, color))
        elif (mm := _TEXT_RE.fullmatch(s)):
            x, y, fs, weight, family, color, text = mm.groups()
            prims.append(('text', (float(x), float(y), float(fs)), text,
                          (color, weight, family)))
        else:
            mm = _RECT_RE.fullmatch(s)
            x, y, h, w, fill, stroke, sw = mm.groups()
            x, y, h, w, sw = map(float, (x, y, h, w, sw))
            prims.append(('rect', (x, y, x+w, y+h), None,
                          (fill, stroke, sw)))
    return prims


def _mpl_prims(cmds):
    prims = []
    for c in mpl_logo_basic._flatten(cmds):
        if isinstance(c, mpl_logo_basic.TextInBox):
            (x0, y0), (x1, y1) = c.upper_left, c.lower_right
            prims.append(('box', (x0, y0, x1, y1), c.text, c.color))
        elif isinstance(c, mpl_logo_basic.Text):
            prims.append(('text', (c.lower_left[0], c.lower_left[1], c.fontsize),
                          c.text, (c.color, c.font_weight, c.font_family)))
        else:
            (x0, y0), (x1, y1) = c.upper_left, c.lower_right
            prims.append(('rect', (x0, y0, x1, y1), None,
                          (c.fill, c.stroke, float(c.stroke_width))))
    return prims


def _assert_same_prims(svg_prims, mpl_prims):
    assert len(svg_prims) == len(mpl_prims) > 0
    for p, q in zip(svg_prims, mpl_prims):
        assert p[0] == q[0], (p, q)
        assert p[2] == q[2], (p, q) # glyph text
        if p[0] == 'text':
            assert p[3] == q[3], (p, q)
        else:
            assert p[3] == q[3], (p, q) # colors / fill,stroke,width
        # svg coords are written with 3-6 decimals
        np.testing.assert_allclose(p[1], q[1], rtol=1e-4, atol=2e-3,
                                   err_msg=f'{p} {q}')


def _count_svg_elements(svg_text):
    return svg_text.count('<text ') + svg_text.count('<rect ')


## same geometry from both drawers #############################################

@pytest.mark.parametrize('tcr_set', sorted(TCR_SETS))
@pytest.mark.parametrize('chain', ['A', 'B'])
def test_tcr_logo_drawers_get_identical_geometry(tcr_set, chain,
                                                 tcrdist_calculator):
    tcrs = TCR_SETS[tcr_set]()
    upper_left, wh = [xmargin, ymargin], (default_width, default_height)
    svg_cmds = make_tcr_logo_svg_commands_for_tcrs(
        tcrs, chain, 'human', upper_left, wh,
        tcrdist_calculator=tcrdist_calculator)
    mpl_cmds = make_tcr_logo_svg_commands_for_tcrs(
        tcrs, chain, 'human', upper_left, wh,
        tcrdist_calculator=tcrdist_calculator, drawer=mpl_logo_basic)

    svg_text = '\n'.join(svg_cmds)
    svg_prims = _parse_svg(svg_text)
    assert len(svg_prims) == _count_svg_elements(svg_text) # parsed them all
    mpl_prims = _mpl_prims(mpl_cmds)
    _assert_same_prims(svg_prims, mpl_prims)

    kinds = {p[0] for p in mpl_prims}
    assert kinds == {'box', 'text', 'rect'} # V/J stacks, CDR3, labels, bars
    if tcr_set == 'diverse':
        assert any(p[2] == '-' for p in mpl_prims) # a gap column was drawn


@pytest.mark.parametrize('signcolors', [False, True])
def test_rank_genes_logo_drawers_get_identical_geometry(signcolors):
    ranks = [('KLRB1', 3.1, 1e-12), ('CEBPD', 1.5, 1e-7), ('KLRG1', 0.4, 2e-5),
             ('NKG7', -1.2, 3e-3), ('GZMK', 2.2, 0.2)]
    kw = dict(top_pval_for_max_height=1e-6, signcolors=signcolors)
    svg_cmds = plotting.make_rank_genes_logo_stack(
        ranks, [10, 10], 800, 1000, **kw)
    mpl_cmds = plotting.make_rank_genes_logo_stack(
        ranks, [10, 10], 800, 1000, drawer=mpl_logo_basic, **kw)
    svg_prims = _parse_svg('\n'.join(svg_cmds))
    assert len(svg_prims) == len(svg_cmds) == len(ranks)
    _assert_same_prims(svg_prims, _mpl_prims(mpl_cmds))


def test_lit_matches_logo_drawers_get_identical_geometry():
    lit_matches = pd.DataFrame(dict(
        clone_index=[0, 1, 1, 3, 7],
        db_mhc=['HLA-A*02:01', 'HLA-A*02:01', np.nan, 'HLA-B*08:01', 'X'],
        db_epitope=['GILGFVFTL', 'GILGFVFTL', 'NLVPMVATV', np.nan, 'ELAGIGILTV']))
    args = ([0, 1, 2, 3], lit_matches)
    svg_cmds = plotting._lit_matches_logo_cmds(*args, svg_basic, 10, 950, 1000)
    mpl_cmds = plotting._lit_matches_logo_cmds(*args, mpl_logo_basic, 10, 950, 1000)
    svg_prims = _parse_svg('\n'.join(svg_cmds))
    assert len(svg_prims) == 3 # A02-GIL, XXX-NLV, B08-XXX
    _assert_same_prims(svg_prims, _mpl_prims(mpl_cmds))
    assert plotting._lit_matches_logo_cmds(
        [4, 5], lit_matches, mpl_logo_basic, 10, 950, 1000) is None


## matplotlib rendering follows the SVG conventions #############################

def _glyph_extents(cmd):
    (path, color), = mpl_logo_basic._patches_for_cmd(cmd)
    return path.get_extents(), color


def test_text_in_box_scaling_matches_svg_convention():
    # 'E': no descender, flat top and bottom. In the SVG convention a 100-pt
    # glyph is 60 wide x 75 tall with its baseline on the box bottom.
    x0, y0, w, h = 100., 200., 60., 75.
    ext, color = _glyph_extents(mpl_logo_basic.text_in_box(
        (x0, y0), (x0+w, y0+h), 'E', 'red'))
    assert color == 'red'
    assert ext.y1 == pytest.approx(y0+h, abs=1e-6) # baseline at box bottom
    # DejaVu Sans Mono cap height is 0.729 em; 0.729/0.75 of the box height
    assert (y0+h - ext.y0) / h == pytest.approx(0.729/0.75, abs=0.01)
    assert x0 < ext.x0 < ext.x1 < x0 + w

    # x and y scale independently; anchored at (x0, box bottom)
    ext2, _ = _glyph_extents(mpl_logo_basic.text_in_box(
        (x0, y0-75), (x0+3*w, y0+h), 'E', 'red'))
    assert (ext2.x0 - x0) == pytest.approx(3*(ext.x0 - x0))
    assert (ext2.x1 - x0) == pytest.approx(3*(ext.x1 - x0))
    assert (y0+h - ext2.y0) == pytest.approx(2*(y0+h - ext.y0))

    # multi-character text is squeezed into the same box; height unchanged
    ext3, _ = _glyph_extents(mpl_logo_basic.text_in_box(
        (x0, y0), (x0+w, y0+h), 'EE', 'red'))
    assert x0 < ext3.x0 < ext.x0 # first E is half as wide
    assert ext.x1 < ext3.x1 < x0 + w
    assert (ext3.y0, ext3.y1) == pytest.approx((ext.y0, ext.y1))


def _render_cmds(cmds, width, height):
    ''' rasterize at 1 pixel per svg unit; returns rgb array indexed [y,x]
    with y increasing downward (svg coords)
    '''
    fig = plt.figure(figsize=(width/100., height/100.), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    mpl_logo_basic.draw_cmds(ax, cmds, width, height)
    ax.axis('off')
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] / 255.
    plt.close(fig)
    assert img.shape == (height, width, 3)
    return img


def test_rect_stroke_is_centered_on_edge_like_svg():
    patches = mpl_logo_basic._patches_for_cmd(mpl_logo_basic.rectangle(
        (10, 20), (110, 70), 'none', 'black', stroke_width=4))
    (ring, color), = patches # no fill
    assert color == 'black'
    ext = ring.get_extents()
    assert (ext.x0, ext.y0, ext.x1, ext.y1) == pytest.approx((8, 18, 112, 72))

    img = _render_cmds([mpl_logo_basic.rectangle(
        (10, 20), (110, 70), 'none', 'black', stroke_width=4)], 120, 80)
    black, white = np.zeros(3), np.ones(3)
    for x, y in [(9, 45), (11, 45), (60, 19), (60, 21), (111, 45), (60, 71)]:
        np.testing.assert_allclose(img[y, x], black, atol=0.01) # the stroke
    for x, y in [(60, 45), (13, 45), (60, 23), (6, 45), (60, 74)]:
        np.testing.assert_allclose(img[y, x], white, atol=0.01) # hole, outside

    (fill, fcolor), (stroke, scolor) = mpl_logo_basic._patches_for_cmd(
        mpl_logo_basic.rectangle((0, 0), (10, 5), 'red', 'red'))
    assert fcolor == scolor == 'red'
    assert fill.contains_point((5, 2.5))


def test_glyph_holes_render_as_holes():
    # 'O' drawn big: the counter (center) must be unpainted
    img = _render_cmds([mpl_logo_basic.text_in_box(
        (0, 0), (60, 75), 'O', 'black')], 60, 80)
    np.testing.assert_allclose(img[38, 30], np.ones(3), atol=0.01) # center
    assert img[38, 6].max() < 0.1 # left side of the O is inked


def test_draw_tcr_logo_for_tcrs_canvas(tcrdist_calculator):
    fig = plt.figure(figsize=(8, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_tcr_logo_for_tcrs(ax, TCR_SETS['trav1-2'](), 'A', 'human',
                           tcrdist_calculator=tcrdist_calculator)
    assert ax.get_xlim() == (0, 2*xmargin + default_width)
    assert ax.get_ylim() == (2*ymargin + default_height, 0) # y downward, as svg
    assert len(ax.patches) > 50
    fig.canvas.draw() # renders without error
    plt.close(fig)


def test_unknown_drawing_command_fails_loudly():
    fig, ax = plt.subplots()
    with pytest.raises(AssertionError):
        mpl_logo_basic.draw_cmds(ax, ['<text>not a command</text>'], 10, 10)
    plt.close(fig)
