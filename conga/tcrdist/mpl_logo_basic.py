''' Matplotlib versions of the logo-drawing primitives in tcrdist_svg_basic.py
(and conga/svg_basic.py), so that logos can be drawn straight onto a
matplotlib Axes instead of going SVG file -> external converter -> PNG -> imread.

Each primitive here has the same signature as its SVG counterpart and uses the
same SVG-style coordinate system (units of SVG "pixels", y increasing
downward), but returns a small drawing-command tuple instead of an SVG string.
That way the layout code (e.g. make_tcr_logo.make_tcr_logo) can be shared
verbatim between the two renderers by passing in `drawer=mpl_logo_basic`
instead of `drawer=tcrdist_svg_basic`. Use draw_cmds to render the commands.

Glyph geometry reproduces the SVG text_in_box convention exactly: a font-size
100 monospace character is taken to be 60 units wide and 75 units tall
(all_text_width/all_text_height in tcrdist_svg_basic), its baseline sits on
the bottom edge of the box, and it is scaled independently in x and y to fill
the box. Text is drawn as filled glyph outlines (TextPath) from the DejaVu Sans
Mono font that ships with matplotlib, so the output does not depend on system
fonts. Rectangle strokes are drawn as filled outlines in data units (as in SVG)
rather than as matplotlib lines, so they scale with the logo.
'''
from collections import namedtuple
from functools import lru_cache

import numpy as np
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.textpath import TextPath

from .basic import MONOSPACE_FONT_FAMILY
from .tcrdist_svg_basic import (all_text_height, all_text_width,
                                protein_logo_boxes, stack_boxes,
                                tree_line_geometry, tree_text_lower_left)

# drawing commands, in SVG coordinates (y increases downward)
TextInBox = namedtuple('TextInBox', 'upper_left lower_right text color')
Text = namedtuple('Text', 'text lower_left fontsize font_family color font_weight')
Rect = namedtuple('Rect', 'upper_left lower_right fill stroke stroke_width dashed')
Line = namedtuple('Line', 'p0 p1 stroke_width color') # butt-capped, like svg <line>

# svg stroke-dasharray used by tcrdist_svg_basic.rectangle(..., dashed=True)
_DASH_ON, _DASH_OFF = 5.0, 5.0

# width and height of a monospace character, in units of the font size,
# as assumed by the SVG text_in_box code (which uses font-size 100)
_CHAR_WIDTH = all_text_width[100] / 100.
_CHAR_HEIGHT = all_text_height[100] / 100.


def _font_properties(font_family, font_weight):
    assert font_weight in ['normal', 'bold']
    prop = FontProperties(family=font_family, weight=font_weight)
    # fail loudly instead of silently substituting a different font
    findfont(prop, fallback_to_default=False)
    return prop

_font_properties(MONOSPACE_FONT_FAMILY, 'normal') # check at import time


@lru_cache(maxsize=None)
def _glyph_path(text, font_family, font_weight):
    ''' Returns (vertices, codes) for text at font size 1, baseline at y=0,
    y increasing upward; None if there is nothing to draw
    '''
    if not text.strip(): # TextPath fails on ''; svg draws nothing
        return None
    path = TextPath((0, 0), text, size=1,
                    prop=_font_properties(font_family, font_weight),
                    usetex=False)
    return np.array(path.vertices), np.array(path.codes)


## the primitives, same signatures as in tcrdist_svg_basic #####################

def text_in_box( upper_left, lower_right, text, color ):
    return TextInBox(tuple(upper_left), tuple(lower_right), text, color)


def make_text( text, lower_left, fontsize, extra_tag = None,
               font_family = MONOSPACE_FONT_FAMILY, color = "black",
               font_weight = "normal" ):
    assert font_weight in ['normal','bold']
    return Text(text, tuple(lower_left), fontsize, font_family, color,
                font_weight)


def rectangle( upper_left, lower_right, fill, stroke, stroke_width=1,
               dashed=False ):
    return Rect(tuple(upper_left), tuple(lower_right), fill, stroke,
                stroke_width, bool(dashed))


def protein_logo( upper_left, lower_right, pwm, scale={} ):
    return [text_in_box(p0, p1, aa, color) for p0, p1, aa, color in
            protein_logo_boxes(upper_left, lower_right, pwm, scale)]


def make_stack( upper_left, lower_right, l ):
    return [text_in_box(p0, p1, word, color) for p0, p1, word, color in
            stack_boxes(upper_left, lower_right, l)]


class MPL_tree_plotter:
    ''' Same interface as tcrdist_svg_basic.SVG_tree_plotter, which is what
    score_trees_devel.Canvas_tree draws the tree with
    '''
    def __init__( self, cmap=None ):
        self.cmds = []
        self.cmap = cmap

    def make_line( self, p0, p1, line_width, normalized_score, extra_tag=None,
                   color=None ):
        geom = tree_line_geometry(p0, p1, line_width, normalized_score,
                                  color=color, cmap=self.cmap)
        if geom[0] == 'rect':
            _, upper_left, lower_right, color = geom
            self.cmds.append(rectangle(upper_left, lower_right, color, color))
        else:
            _, p0, p1, stroke_width, color = geom
            self.cmds.append(Line(tuple(p0), tuple(p1), stroke_width, color))

    def make_text( self, text, p, fontsize, extra_tag=None ):
        self.cmds.append(make_text(text, tree_text_lower_left(text, p, fontsize),
                                   fontsize, color='black'))


def make_tree_plotter( cmap=None ):
    ''' see tcrdist_svg_basic.make_tree_plotter '''
    return MPL_tree_plotter( cmap=cmap )


## rendering ###################################################################

def _flatten(cmds):
    for cmd in cmds:
        if isinstance(cmd, (TextInBox, Text, Rect, Line)):
            yield cmd
        else:
            assert isinstance(cmd, list), f'unrecognized drawing command: {cmd!r}'
            yield from _flatten(cmd)


def _rect_vertices(x0, y0, x1, y1, reverse=False):
    verts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if reverse:
        verts.reverse()
    return verts + [verts[0]]


def _closed_path(*vertex_lists):
    verts, codes = [], []
    for vl in vertex_lists:
        verts.extend(vl)
        codes.extend([Path.MOVETO] + [Path.LINETO]*(len(vl)-2) + [Path.CLOSEPOLY])
    return Path(verts, codes)


def _thick_segment_vertices(p, q, w):
    ''' polygon for a segment of width w with butt ends, as svg strokes '''
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    length = np.sqrt(np.sum(d**2))
    assert length > 0
    n = np.array([-d[1], d[0]]) * (w/2) / length
    verts = [p+n, q+n, q-n, p-n]
    return [tuple(x) for x in verts] + [tuple(verts[0])]


def _dashed_rect_stroke_vertex_lists(x0, y0, x1, y1, w):
    ''' svg stroke-dasharray:5,5 around the rect outline: the dash pattern
    runs continuously along the path (x0,y0)->(x1,y0)->(x1,y1)->(x0,y1)->(x0,y0)
    with butt caps; a dash that spans a corner gets the mitered corner square
    '''
    corners = [(x0,y0), (x1,y0), (x1,y1), (x0,y1), (x0,y0)]
    seglens = [x1-x0, y1-y0, x1-x0, y1-y0]
    perimeter = sum(seglens)
    vertex_lists = []
    a = 0.0
    while a < perimeter:
        b = min(a + _DASH_ON, perimeter)
        s0 = 0.0
        for i in range(4):
            s1 = s0 + seglens[i]
            lo, hi = max(a, s0), min(b, s1)
            if hi > lo and seglens[i] > 0:
                (ax, ay), (bx, by) = corners[i], corners[i+1]
                f0, f1 = (lo-s0)/seglens[i], (hi-s0)/seglens[i]
                vertex_lists.append(_thick_segment_vertices(
                    (ax + (bx-ax)*f0, ay + (by-ay)*f0),
                    (ax + (bx-ax)*f1, ay + (by-ay)*f1), w))
            # corner join inside this dash (the start corner: the first dash
            # always starts there, so join if this dash runs to the end)
            if (a < s1 < b) or (i == 3 and b == perimeter):
                cx, cy = corners[i+1]
                vertex_lists.append(_rect_vertices(cx-w/2, cy-w/2, cx+w/2, cy+w/2))
            s0 = s1
        a += _DASH_ON + _DASH_OFF
    return vertex_lists


def _patches_for_cmd(cmd):
    ''' Returns a list of (path, color) in SVG coordinates
    '''
    if isinstance(cmd, TextInBox):
        glyph = _glyph_path(cmd.text, MONOSPACE_FONT_FAMILY, 'normal')
        if glyph is None:
            return []
        verts, codes = glyph
        (x0, y0), (x1, y1) = cmd.upper_left, cmd.lower_right
        # same scaling as the SVG version: transform=scale(x_scale,y_scale)
        # applied to a font-size-100 text element with baseline at y1
        x_scale = (x1-x0) / (_CHAR_WIDTH * len(cmd.text))
        y_scale = (y1-y0) / _CHAR_HEIGHT
        new_verts = np.column_stack([x0 + x_scale * verts[:,0],
                                     y1 - y_scale * verts[:,1]])
        return [(Path(new_verts, codes), cmd.color)]

    elif isinstance(cmd, Text):
        glyph = _glyph_path(cmd.text, cmd.font_family, cmd.font_weight)
        if glyph is None:
            return []
        verts, codes = glyph
        x0, y0 = cmd.lower_left
        new_verts = np.column_stack([x0 + cmd.fontsize * verts[:,0],
                                     y0 - cmd.fontsize * verts[:,1]])
        return [(Path(new_verts, codes), cmd.color)]

    elif isinstance(cmd, Line):
        return [(_closed_path(_thick_segment_vertices(cmd.p0, cmd.p1,
                                                      cmd.stroke_width)),
                 cmd.color)]

    else:
        assert isinstance(cmd, Rect)
        (x0, y0), (x1, y1) = cmd.upper_left, cmd.lower_right
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        patches = []
        if cmd.fill != 'none':
            patches.append((_closed_path(_rect_vertices(x0, y0, x1, y1)),
                            cmd.fill))
        w = cmd.stroke_width
        if cmd.stroke != 'none' and w > 0 and cmd.dashed:
            for vl in _dashed_rect_stroke_vertex_lists(x0, y0, x1, y1, w):
                patches.append((_closed_path(vl), cmd.stroke))
        elif cmd.stroke != 'none' and w > 0:
            # SVG strokes are centered on the edge
            outer = _rect_vertices(x0-w/2, y0-w/2, x1+w/2, y1+w/2)
            if x1-x0 > w and y1-y0 > w: # a ring, with a hole
                inner = _rect_vertices(x0+w/2, y0+w/2, x1-w/2, y1-w/2,
                                       reverse=True)
                patches.append((_closed_path(outer, inner), cmd.stroke))
            else: # stroke covers the whole interior
                patches.append((_closed_path(outer), cmd.stroke))
        return patches


def draw_cmds( ax, cmds, width, height ):
    ''' Draw the commands onto ax, set up to show the SVG canvas
    [0,width] x [0,height] (y increasing downward) with equal aspect ratio,
    the way plt.imshow showed the converted PNG.

    Commands are drawn in order, so later ones are on top, as in SVG.
    '''
    assert width > 0 and height > 0
    for cmd in _flatten(cmds):
        for path, color in _patches_for_cmd(cmd):
            ax.add_patch(PathPatch(path, facecolor=color, edgecolor='none',
                                   linewidth=0))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')


def _cmds_figure( cmds, width, height ):
    ''' a Figure (1 inch = 100 svg units) showing the width x height canvas '''
    width, height = int(width), int(height) # as in the svg header
    fig = Figure(figsize=(width/100., height/100.), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_cmds(ax, cmds, width, height)
    ax.axis('off')
    return fig


def save_cmds_as_png( cmds, width, height, pngfile, scale=1.0 ):
    ''' Render cmds on a canvas of width x height svg units and save as a png
    with a white background, with `scale` png pixels per svg unit. At the
    default scale=1 the png is the same size as the one the svg->png converters
    made from tcrdist_svg_basic.create_file(cmds, width, height, ...); larger
    scales give the same layout at higher resolution.
    '''
    assert pngfile.endswith('.png')
    assert scale > 0
    _cmds_figure(cmds, width, height).savefig(
        pngfile, dpi=100*scale, facecolor='white')


def save_cmds_as_pdf( cmds, width, height, pdffile ):
    ''' Vector (pdf) version of save_cmds_as_png; text is drawn as glyph
    outlines, so no fonts need to be embedded
    '''
    assert pdffile.endswith('.pdf')
    _cmds_figure(cmds, width, height).savefig(pdffile, facecolor='white')
