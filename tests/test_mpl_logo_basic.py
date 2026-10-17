"""Tests for conga/tcrdist/mpl_logo_basic.py, the matplotlib renderer for the
logos in the graph-vs-graph / tcr-clumping figures (svg_free branch).

The layout code (make_tcr_logo, make_rank_genes_logo_stack, ...) is shared by
the legacy SVG renderer and the matplotlib renderer via a `drawer` argument.
These tests check:
  * that, for real TCR logos, DEG/TCR-feature logos, and TCRdist trees, the two
    drawers are handed exactly the same boxes, glyphs, rectangles and colors,
    in the same order (parsed back out of the SVG text), and
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
from conga.tcrdist.make_tcr_trees import make_tcr_tree_svg_commands
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
_TEXT_RE = re.compile( # make_text; tree plotter text has no font-weight
    r'<text x="([-\d.]+)" y="([-\d.]+)" font-size="([-\d.e]+)" '
    r'(?:font-weight="(\w+)" )?font-family="([^"]+)" fill="([^"]+)" '
    r'xml:space="preserve">([^<]*)</text>')
_RECT_RE = re.compile(
    r'<rect x="([-\d.e]+)" y="([-\d.e]+)" height="([-\d.e]+)" width="([-\d.e]+)" '
    r'style="fill:([^;]+);stroke:([^;]+);stroke-width:([-\d.e]+)'
    r'(;stroke-dasharray:5,5)?" />')
_LINE_RE = re.compile(
    r'<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)" '
    r'stroke-width="([-\d.e]+)" stroke="([^"]+)"/>')


def _parse_svg(svg_text):
    ''' Returns list of primitives in document order, in the same form as the
    mpl_logo_basic commands
    '''
    prims = []
    pattern = re.compile('|'.join(
        f'(?:{r.pattern})' for r in [_TEXT_IN_BOX_RE, _TEXT_RE, _RECT_RE,
                                     _LINE_RE]))
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
                          (color, weight or 'normal', family)))
        elif (mm := _RECT_RE.fullmatch(s)):
            x, y, h, w, fill, stroke, sw, dashed = mm.groups()
            x, y, h, w, sw = map(float, (x, y, h, w, sw))
            prims.append(('rect', (x, y, x+w, y+h), None,
                          (fill, stroke, sw, dashed is not None)))
        else:
            mm = _LINE_RE.fullmatch(s)
            x1, y1, x2, y2, sw, color = mm.groups()
            prims.append(('line', tuple(map(float, (x1, y1, x2, y2, sw))), None,
                          color))
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
        elif isinstance(c, mpl_logo_basic.Line):
            prims.append(('line', (*c.p0, *c.p1, c.stroke_width), None, c.color))
        else:
            (x0, y0), (x1, y1) = c.upper_left, c.lower_right
            prims.append(('rect', (x0, y0, x1, y1), None,
                          (c.fill, c.stroke, float(c.stroke_width), c.dashed)))
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
    return (svg_text.count('<text ') + svg_text.count('<rect ') +
            svg_text.count('<line '))


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


## TCRdist trees ###############################################################

def _tree_inputs(tcrdist_calculator):
    # MAIT-like TCRs form clusters (--> logos, dashed boxes); others add diversity
    tcrs = (_load_tcrs(lambda df: df.va_gene.str.startswith('TRAV1-2*'), 40) +
            _load_tcrs(lambda df: ~df.va_gene.str.startswith('TRAV1-2*'), 40))
    D = np.array([[tcrdist_calculator(a, b) for b in tcrs] for a in tcrs])
    scores = list(np.linspace(0, 3, len(tcrs)))
    return tcrs, D, scores


def test_tree_drawers_get_identical_geometry(tcrdist_calculator):
    tcrs, D, scores = _tree_inputs(tcrdist_calculator)
    kw = dict(tcrdist_calculator=tcrdist_calculator, color_scores=scores,
              color_score_range=[0, 3.0], title='test tree')
    svg_cmds = make_tcr_tree_svg_commands(
        tcrs, 'human', [25, 0], [800, 1000], D, **kw)
    mpl_cmds = make_tcr_tree_svg_commands(
        tcrs, 'human', [25, 0], [800, 1000], D, drawer=mpl_logo_basic, **kw)

    svg_text = '\n'.join(svg_cmds)
    svg_prims = _parse_svg(svg_text)
    assert len(svg_prims) == _count_svg_elements(svg_text) # parsed them all
    mpl_prims = _mpl_prims(mpl_cmds)
    _assert_same_prims(svg_prims, mpl_prims)

    # exercised: tree branches, dashed glyph boxes, labels, and TCR logos
    rects = [p for p in mpl_prims if p[0] == 'rect']
    assert sum(p[3][3] for p in rects) >= 2 # dashed boxes around subtrees
    assert sum(not p[3][3] for p in rects) > 100 # branches, junction bars
    assert any(p[0] == 'text' and p[2] == 'test tree' for p in mpl_prims)
    assert any(p[0] == 'box' and p[2] == 'S' for p in mpl_prims) # logo glyphs


def test_dashed_rect_follows_svg_dash_pattern():
    # stroke-dasharray:5,5 runs along the outline from the upper left corner:
    # top edge left-to-right (s=0..50), then the right edge downward (s=50..80)
    img = _render_cmds([mpl_logo_basic.rectangle(
        (10, 10), (60, 40), 'none', 'black', stroke_width=3, dashed=True)],
                       80, 60)
    def inked(x, y):
        return img[y, x].max() < 0.1
    def blank(x, y):
        return img[y, x].min() > 0.9
    for s in [1, 3, 11, 13, 21, 43]: # dashes on [0,5), [10,15), ...
        assert inked(10 + s, 10), s
    for s in [6, 8, 16, 18, 37]: # gaps on [5,10), [15,20), ...
        assert blank(10 + s, 10), s
    assert inked(60, 10 + 2) and blank(60, 10 + 7) and inked(60, 10 + 12)
    assert inked(60, 10) # corner inside the dash at s=[50,55) is filled (miter)
    assert blank(35, 25) # interior unfilled


def test_tree_plotter_text_is_right_aligned_like_svg():
    plotter = mpl_logo_basic.make_tree_plotter()
    plotter.make_text('150', (200, 50), 10)
    cmd, = plotter.cmds
    assert cmd.lower_left == pytest.approx((200 - 10*0.6*3, 50))
    assert (cmd.fontsize, cmd.color, cmd.font_weight) == (10, 'black', 'normal')


def test_tree_plotter_lines():
    plotter = mpl_logo_basic.make_tree_plotter()
    plotter.make_line([10, 20], [10, 80], 4, 0.5)        # vertical
    plotter.make_line([10, 20], [90, 20], 6, None)       # horizontal, no score
    plotter.make_line([0, 0], [30, 40], 4, 1.0)          # diagonal
    v, h, d = plotter.cmds
    assert (v.upper_left, v.lower_right) == ((8, 20), (12, 80))
    assert v.fill == v.stroke == tcrdist_svg_basic.rgb_from_fraction(0.5)
    assert (h.upper_left, h.lower_right) == ((10, 17), (90, 23))
    assert h.fill == '#aaaaaa'
    assert isinstance(d, mpl_logo_basic.Line) and d.stroke_width == 2
    (path, color), = mpl_logo_basic._patches_for_cmd(d)
    ext = path.get_extents() # 2-wide segment from (0,0) to (30,40), butt caps
    assert (ext.x0, ext.x1) == pytest.approx((-0.8, 30.8))
    assert (ext.y0, ext.y1) == pytest.approx((-0.6, 40.6))


def test_save_cmds_as_png_canvas_size(tmp_path):
    pngfile = str(tmp_path / 'test.png')
    mpl_logo_basic.save_cmds_as_png(
        [mpl_logo_basic.rectangle((0, 0), (50, 20), 'red', 'red')], 230, 170,
        pngfile)
    img = plt.imread(pngfile)
    assert img.shape[:2] == (170, 230) # one svg unit per pixel
    np.testing.assert_allclose(img[10, 25, :3], [1, 0, 0], atol=0.01)
    np.testing.assert_allclose(img[100, 100, :3], [1, 1, 1], atol=0.01)

    # higher resolution: same layout, 3 pixels per svg unit
    mpl_logo_basic.save_cmds_as_png(
        [mpl_logo_basic.rectangle((0, 0), (50, 20), 'red', 'red')], 230, 170,
        pngfile, scale=3)
    img3 = plt.imread(pngfile)
    assert img3.shape[:2] == (3*170, 3*230)
    np.testing.assert_allclose(img3[3*10, 3*25, :3], [1, 0, 0], atol=0.01)
    np.testing.assert_allclose(img3[3*20+3, 3*25, :3], [1, 1, 1], atol=0.01) # below rect+stroke


def test_empty_text_draws_nothing():
    # svg draws nothing for empty <text>; matplotlib's TextPath fails on ''
    assert mpl_logo_basic._patches_for_cmd(mpl_logo_basic.make_text('', (0, 0), 10)) == []
    assert mpl_logo_basic._patches_for_cmd(
        mpl_logo_basic.text_in_box((0, 0), (5, 5), '', 'black')) == []
