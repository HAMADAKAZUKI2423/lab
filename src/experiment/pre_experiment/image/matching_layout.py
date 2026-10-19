"""上下の角度範囲を各平面へ射影。中央境界は検査、画面外は警告のみ。"""
from dataclasses import dataclass, asdict
import math

@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    @property
    def size(self): return (self.x1-self.x0, self.y1-self.y0)
    @property
    def center(self): return ((self.x0+self.x1)/2, (self.y0+self.y1)/2)
    def contains(self, width, height):
        return self.x0 >= 0 and self.y0 >= 0 and self.x1 <= width and self.y1 <= height
    def translated(self, dx, dy):
        return Box(self.x0+dx, self.y0+dy, self.x1+dx, self.y1+dy)
    def intersects(self, other):
        return max(self.x0,other.x0) < min(self.x1,other.x1) and max(self.y0,other.y0) < min(self.y1,other.y1)

def project_box(width, height, distance, ppcm, center_deg, vertical_deg, horizontal_deg, dx=0, dy=0):
    """角度上の中心±半角をtanで射影。矩形画素中心と角度中心を混同しない。"""
    if min(width,height,distance,ppcm,vertical_deg,horizontal_deg) <= 0:
        raise ValueError("Invalid canvas or geometry")
    upper=center_deg+vertical_deg/2; lower=center_deg-vertical_deg/2
    if max(abs(upper),abs(lower),horizontal_deg/2) >= 85:
        raise ValueError("Angular range is outside the projection model")
    cx=width//2+dx; cy=height//2+dy
    half=distance*math.tan(math.radians(horizontal_deg/2))*ppcm
    return Box(round(cx-half),round(cy-distance*math.tan(math.radians(upper))*ppcm),
               round(cx+half),round(cy-distance*math.tan(math.radians(lower))*ppcm))

def make_layout(config, foreground_size, background_size, offset_x=0, offset_y=0):
    fw,fh=foreground_size; bw,bh=background_size
    common=(config.visual_angle_deg, config.visual_angle_deg)
    ref=project_box(fw,fh,config.distance_fg_cm,config.foreground_pixels_per_cm,
                    config.reference_center_deg,*common)
    test=project_box(fw,fh,config.distance_fg_cm,config.foreground_pixels_per_cm,
                    config.test_center_deg,*common)
    sp=project_box(fw,fh,config.distance_fg_cm,config.foreground_pixels_per_cm,
                  config.test_center_deg,config.visual_angle_deg,config.background_visual_angle_width_deg)
    bg=project_box(bw,bh,config.distance_bg_cm,config.background_pixels_per_cm,
                  config.test_center_deg,config.visual_angle_deg,config.background_visual_angle_width_deg,
                  offset_x,offset_y)
    # BGの正方形校正枠も下段。横長背景枠と同じ上下端を使う。
    bg_square=project_box(bw,bh,config.distance_bg_cm,config.background_pixels_per_cm,
                         config.test_center_deg,*common,offset_x,offset_y)
    ref_bg=project_box(bw,bh,config.distance_bg_cm,config.background_pixels_per_cm,
                      config.reference_center_deg,config.visual_angle_deg,
                      config.background_visual_angle_width_deg,offset_x,offset_y)
    ref_sp=project_box(fw,fh,config.distance_fg_cm,config.foreground_pixels_per_cm,
                      config.reference_center_deg,config.visual_angle_deg,
                      config.background_visual_angle_width_deg)
    return {'reference':ref,'reference_singleplane':ref_sp,'reference_background':ref_bg,'test':test,'singleplane':sp,'background':bg,'background_square':bg_square}

def validate_layout(layout, foreground_size, background_size, singleplane):
    """画面外の刺激は警告のみ。サイズ・比率・位置は変えずCanvasで見切れる。"""
    fg_names=('reference_singleplane','singleplane' if singleplane else 'test')
    checks=[(name,foreground_size) for name in fg_names]
    if not singleplane:
        checks.append(('background',background_size))
    for name,size in checks:
        if not layout[name].contains(*size):
            print(
                f"WARN: {name} extends outside the display: {layout[name]}, "
                f"canvas={size}. Continuing without resizing or repositioning; "
                "outside pixels are not visible.",
                flush=True,
            )
    # 画面内収容とは別の検査。上下刺激の重なりと中央境界は維持する。
    if layout['reference_singleplane'].intersects(layout[fg_names[1]]):
        raise ValueError("Reference and test overlap")
    if layout['reference'].y1 != layout['test'].y0:
        raise ValueError("Central boundary has a gap or overlap")

def layout_metadata(layout):
    return {name:asdict(box) for name,box in layout.items()}

def draw_brackets(canvas, box, color, width=5):
    x0,y0,x1,y1=box.x0,box.y0,box.x1,box.y1; length=min(30,*box.size)//2
    for x,y,sx,sy in ((x0,y0,1,1),(x1,y0,-1,1),(x0,y1,1,-1),(x1,y1,-1,-1)):
        canvas.create_line(x,y,x+sx*length,y,fill=color,width=width,tags='calib')
        canvas.create_line(x,y,x,y+sy*length,fill=color,width=width,tags='calib')
