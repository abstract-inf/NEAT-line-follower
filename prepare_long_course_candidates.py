"""Prepare five performance-blind long-course options; never run a controller."""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from line_follower.tracks import Branch, Section, TrackMask, TrackSpec, branch_points, sample_route
from run_corner_training import protected_tracks

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs/long-course-candidates-20261121"


def straight(length):
    return Section("straight", length=length)


def smooth_s(radius, sign=1):
    return [Section("arc", radius=radius, angle=sign*math.radians(a), feature="s_curve") for a in (45,-90,45)]


def wide_wave(radius, sign=1):
    return [Section("arc", radius=radius, angle=sign*math.radians(a), feature="gentle_curve") for a in (30,-30,-30,30)]


def chicane(turns, step, sign=1):
    """Balanced alternating diagonal legs, with zero net lateral displacement."""
    angles = [45] + [(-90 if i%2==0 else 90) for i in range(turns-2)] + [-45]
    result = []
    for i, angle in enumerate(angles):
        result.append(Section("corner", angle=sign*math.radians(angle), feature="zigzag"))
        if i < turns-1:
            result.append(Section("straight", length=step if i in (0,turns-2) else 2*step, feature="zigzag"))
    return result


def square_steps(size, sign=1):
    result = []
    for i, a in enumerate((90,-90,-90,90)):
        result.append(Section("corner", angle=sign*math.radians(a), feature="corner"))
        if i < 3:
            result.append(Section("straight",length=size,feature="corner"))
    return result


def row(index, total, lead, modules, advance):
    chunks = [("Opening straight" if index==0 else f"Row {index+1} approach", [straight(lead)])]
    chunks.extend(modules)
    route = sample_route(tuple(s for _, sections in chunks for s in sections),(0,0,0))
    heading=sum(s.angle for _,seq in chunks for s in seq if s.kind in ("arc","corner"))
    if abs(route.points[-1,1])>1e-8 or abs(math.remainder(heading,2*math.pi))>1e-8:
        raise ValueError("A lane module must return to its initial lateral position and heading")
    tail = advance-float(route.points[-1,0])
    if tail < .3:
        raise ValueError("Not enough straight clearance after lane modules")
    chunks.append(("Finish straight" if index==total-1 else f"Row {index+1} back straight", [straight(tail)]))
    return chunks


def recipes():
    return [
        {"id":1,"slug":"balanced-grand-prix","title":"Balanced Grand Prix","focus":"Broad curves, braking, zigzags, and re-acceleration in one balanced course.",
         "description":"The recommended all-round case study: a 4.5 m launch, broad S, six-turn chicane, right-angle steps, precision S, and a long finishing straight.",
         "advance":8.8,"hairpins":[.85,.85],
         "rows":[(4.5,[("Broad S · R 1.00 m",smooth_s(1.))]),
                 (1.1,[("Six-turn chicane",chicane(6,.55,-1))]),
                 (2.,[("Right-angle steps",square_steps(.65)),("Precision S · R 0.45 m",smooth_s(.45))])]},
        {"id":2,"slug":"high-speed-sweep","title":"High-speed Sweep","focus":"Sustained speed and gentle curvature, followed by a late braking challenge.",
         "description":"The clearest speed-focused option: a 6 m launch, large-radius sweeps, wide turnarounds, one late sharp chicane, and an almost 6 m finish.",
         "advance":9.4,"hairpins":[1.1,1.1],
         "rows":[(6.,[("Broad S · R 1.00 m",smooth_s(1.))]),
                 (3.,[("Wide wave · R 1.50 m",wide_wave(1.5,-1))]),
                 (1.2,[("Late four-turn chicane",chicane(4,.8))])]},
        {"id":3,"slug":"technical-switchbacks","title":"Technical Switchbacks","focus":"Tight cornering, dense changes of direction, and recovering alignment.",
         "description":"The most technical continuous-line option: four lanes, short-spaced zigzags, right-angle steps, tight turnarounds, and a 0.22 m-radius finishing S.",
         "advance":6.8,"hairpins":[.6,.6,.6],
         "rows":[(3.2,[("Entry four-turn chicane",chicane(4,.45))]),
                 (.8,[("Right-angle steps",square_steps(.5,-1)),("Short reset",[straight(.5)]),("Six-turn chicane",chicane(6,.45,-1))]),
                 (.8,[("Eight-turn chicane",chicane(8,.4))]),
                 (1.,[("Final right-angle steps",square_steps(.55,-1)),("Tight S · R 0.22 m",smooth_s(.22,-1))])]},
        {"id":4,"slug":"rhythm-changes","title":"Rhythm Changes","focus":"Repeated transitions between acceleration, smooth turning, and abrupt steering.",
         "description":"A longer stress course with a broad opening, repeated sharp chicanes, a curved-to-square transition, and a denser late combination.",
         "advance":8.4,"hairpins":[.8,.8,.8],
         "rows":[(4.,[("Flowing S · R 1.10 m",smooth_s(1.1))]),
                 (1.,[("Wide six-turn chicane",chicane(6,.6,-1))]),
                 (1.,[("Curved approach · R 0.70 m",smooth_s(.7)),("Speed reset",[straight(1.2)]),("Right-angle transition",square_steps(.6))]),
                 (1.,[("Four-turn chicane",chicane(4,.45,-1)),("Short reset",[straight(.8)]),("Dense final chicane",chicane(6,.4,-1))])]},
        {"id":5,"slug":"recovery-endurance","title":"Recovery Endurance","focus":"Short line gaps and straight-through distractors embedded in a long mixed course.",
         "description":"The recovery-focused option: three short gaps, two crossing distractors, one side spur, varied bends, and a demanding late combination.",
         "advance":8.8,"hairpins":[.9,.9,.9],
         "rows":[(5.,[("Wide wave · R 1.00 m",wide_wave(1.))]),
                 (1.6,[("Four-turn chicane",chicane(4,.65,-1))]),
                 (2.,[("Smooth S · R 0.80 m",smooth_s(.8))]),
                 (1.,[("Late right-angle steps",square_steps(.6,-1)),("Short reset",[straight(.8)]),("Late six-turn chicane",chicane(6,.5,-1))])]},
    ]


def build(recipe):
    chunks = []
    for i,(lead,modules) in enumerate(recipe["rows"]):
        chunks.extend(row(i,len(recipe["rows"]),lead,modules,recipe["advance"]))
        if i < len(recipe["hairpins"]):
            radius = recipe["hairpins"][i]
            chunks.append((f"Turnaround {i+1} · R {radius:.2f} m",
                           [Section("arc",radius=radius,angle=math.pi*(1 if i%2==0 else -1),feature="u_turn")]))
    sections = tuple(s for _, seq in chunks for s in seq)
    blocks, position = [], 0.
    for name, sequence in chunks:
        length = sum(s.length if s.kind=="straight" else abs(s.radius*s.angle) if s.kind=="arc" else 0. for s in sequence)
        blocks.append({"name":name,"start_m":position,"end_m":position+length,"length_m":length})
        position += length
    track = TrackSpec(recipe["slug"],sections,start=(0.,0.,0.),resolution=.002,arena_size=20.,
                      seed=2026112120+recipe["id"],split="candidate",difficulty="advanced",generator_version="2.0")
    if recipe["id"]==5:
        station = {b["name"]:b for b in blocks}
        g2 = station["Row 3 back straight"]["start_m"]+2.
        g3 = station["Finish straight"]["start_m"]+1.5
        track = replace(track,gaps=((2.5,2.520),(g2,g2+.035),(g3,g3+.045)),
                        branches=(Branch(at=4.,length=.22,crossing=True),
                                  Branch(at=station["Row 2 approach"]["start_m"]+.8,length=.20),
                                  Branch(at=station["Row 3 approach"]["start_m"]+1.,length=.22,crossing=True)))
    route = track.route
    heading = route.heading_at(route.length)
    extent = [route.points, np.array([route.points[-1]+track.runout*np.array([math.cos(heading),math.sin(heading)])])]
    extent.extend(branch_points(track,b) for b in track.branches)
    points = np.vstack(extent)
    lower,upper = points.min(axis=0),points.max(axis=0)
    center = (lower+upper)/2
    arena = math.ceil(float(max(upper-lower))+1.)
    track = replace(track,start=(-float(center[0]),-float(center[1]),0.),arena_size=arena)
    return track,blocks


def check_geometry(track):
    """Same clearance convention as validate_track, with bounded pairwise memory."""
    route = track.route
    if np.any(np.abs(route.points)+route.widths[:,None]/2 > track.arena_size/2-.16):
        raise ValueError("Route leaves arena clearance")
    if any(s.kind=="arc" and s.radius<.12 for s in track.sections):
        raise ValueError("Arc below existing minimum radius")
    if math.ceil(track.arena_size/track.resolution)>8192:
        raise ValueError("Floor raster exceeds existing 8192-pixel limit")
    # The original checker uses a 7-sample stride at 2 mm route sampling.
    points,distance = route.points[::7],route.distance[::7]
    minimum = float("inf")
    for begin in range(0,len(points),48):
        separation = np.linalg.norm(points[begin:begin+48,None]-points[None],axis=2)
        nonlocal_pair = np.abs(distance[begin:begin+48,None]-distance[None])>.45
        if nonlocal_pair.any():
            minimum = min(minimum,float(separation[nonlocal_pair].min()))
    if minimum<.20:
        raise ValueError(f"Unintended route crossing / clearance {minimum:.3f} m")
    heading = route.heading_at(route.length)
    extension = np.linspace(route.points[-1],route.points[-1]+track.runout*np.array([math.cos(heading),math.sin(heading)]),20)
    if np.any(np.abs(extension)+track.sections[-1].width/2>track.arena_size/2-.16):
        raise ValueError("Finish extension leaves arena clearance")
    far = route.points[route.distance<route.length-.45]
    if np.linalg.norm(extension[:,None]-far[None],axis=2).min()<.20:
        raise ValueError("Finish extension approaches earlier paint")
    previous = -1.
    for a,b in track.gaps:
        if not .1<a<b<route.length-.1 or a<previous:
            raise ValueError("Invalid gap")
        previous=b
    for branch in track.branches:
        if not .15<branch.at<route.length-.15 or any(a-.1<branch.at<b+.1 for a,b in track.gaps):
            raise ValueError("Junction at invalid station")
        if any(abs(branch.at-c)<.16 for c in route.corners) or abs(math.remainder(route.heading_at(branch.at+.1)-route.heading_at(branch.at-.1),2*math.pi))>.03:
            raise ValueError("Junction is not safely on a straight")
        line = branch_points(track,branch)
        if np.any(np.abs(line)>track.arena_size/2-.16):
            raise ValueError("Distractor outside arena")
        samples = np.linspace(line[0],line[-1],30)
        far = route.points[np.abs(route.distance-branch.at)>.45]
        if np.linalg.norm(samples[:,None]-far[None],axis=2).min()<.20:
            raise ValueError("Distractor approaches another route segment")
    return minimum


def check_floor(track):
    mask = TrackMask(track)
    stations = np.linspace(.05,track.route.length-.05,5000)
    expected = np.ones(len(stations),dtype=bool)
    for a,b in track.gaps:
        expected &= ~((stations>=a-.01)&(stations<=b+.01))
    points = np.array([track.route.point_at(s) for s in stations[expected]])
    if not np.all(mask.sample(points)==1):
        raise ValueError("Paint missing from intended route outside declared gaps")
    for a,b in track.gaps:
        if mask.sample(np.array([track.route.point_at((a+b)/2)]))[0] != 0:
            raise ValueError("Declared gap is painted")
    return {"pixels":mask.pixels,"meters_per_pixel":mask.meters_per_pixel,
            "retained_grayscale_rasters_mib":round(2*mask.pixels**2/2**20,2),"paint_checks_passed":True}


def draw(track,blocks,axis,details=False):
    route=track.route
    intervals=[]
    cursor=0.
    for a,b in track.gaps:
        intervals.append((cursor,a))
        cursor=b
    intervals.append((cursor,route.length))
    for a,b in intervals:
        samples=np.r_[a,route.distance[(route.distance>a)&(route.distance<b)],b]
        xy=np.column_stack([np.interp(samples,route.distance,route.points[:,i]) for i in (0,1)])
        axis.plot(*xy.T,color="#183d59",linewidth=2.5,solid_capstyle="butt")
    for branch in track.branches:
        axis.plot(*branch_points(track,branch).T,color="#8e5aa7",linewidth=2.1)
        axis.scatter(*route.point_at(branch.at),color="#8e5aa7",s=35,zorder=5,marker="D")
    for a,b in track.gaps:
        axis.scatter(*route.point_at((a+b)/2),facecolors="white",edgecolors="#c43886",s=80,zorder=6,linewidth=1.7)
    if route.corners:
        xy=np.array([route.point_at(s) for s in route.corners])
        axis.scatter(*xy.T,color="#d56a23",s=13,zorder=4)
    axis.scatter(*route.points[0],marker="o",s=60,color="#008b65",edgecolors="white",linewidth=1,zorder=7)
    axis.scatter(*route.points[-1],marker="s",s=60,color="#c34135",edgecolors="white",linewidth=1,zorder=7)
    for s in np.linspace(1.,route.length-1.,8):
        xy=route.point_at(s)
        theta=route.heading_at(s)
        tip=xy+.25*np.array([math.cos(theta),math.sin(theta)])
        axis.annotate("",tip,xy,arrowprops={"arrowstyle":"->","color":"#183d59","lw":1.3},zorder=4)
    axis.annotate("START",route.points[0],xytext=(-5,-15),textcoords="offset points",color="#008b65",weight="bold",fontsize=8)
    axis.annotate("FINISH",route.points[-1],xytext=(4,-15),textcoords="offset points",color="#b5362e",weight="bold",fontsize=8)
    selected=[b for b in blocks if not ("approach" in b["name"] or "back straight" in b["name"] or "reset" in b["name"].lower())]
    if details:
        for i,block in enumerate(selected,1):
            station=(block["start_m"]+block["end_m"])/2
            reserved=[(a+b)/2 for a,b in track.gaps]+[b.at for b in track.branches]
            if any(abs(station-at)<.2 for at in reserved):
                station=min(block["end_m"]-.1,station+min(.4,block["length_m"]/4))
            point=route.point_at(station)
            axis.text(*point,str(i),ha="center",va="center",fontsize=8,color="#183d59",zorder=8,
                      bbox={"boxstyle":"circle,pad=.22","fc":"white","ec":"#183d59","lw":.7})
    axis.set(aspect="equal",xlabel="X (m)",ylabel="Y (m)")
    axis.margins(x=.09,y=.18)
    axis.grid(color="#e7ecf0",linewidth=.6)
    axis.set_axisbelow(True)
    axis.spines[["top","right"]].set_visible(False)
    return selected


def figures(courses,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size":10,"svg.fonttype":"none","axes.titleweight":"bold"})
    fig,axes=plt.subplots(3,2,figsize=(15,13),layout="constrained")
    for (recipe,track,blocks,record),axis in zip(courses,axes.flat):
        draw(track,blocks,axis)
        axis.set_title(f"{recipe['id']}. {recipe['title']}  |  {track.route.length:.1f} m",fontsize=12)
        axis.text(.02,.97,f"{record['sharp_turns']} sharp turns · {record['gaps']} gaps · {record['distractors']} distractors",
                  transform=axis.transAxes,ha="left",va="top",fontsize=9,color="#546778")
    panel=axes.flat[-1]
    panel.axis("off")
    panel.text(.04,.93,"Choose by the question you want to study",fontsize=15,weight="bold",va="top")
    notes=["1  Balanced comparison — recommended","2  Speed and wide curves","3  Dense technical steering","4  Repeated fast-to-slow transitions","5  Gaps, junctions, and recovery"]
    for i,note in enumerate(notes):
        panel.text(.04,.78-i*.12,note,fontsize=12)
    panel.text(.04,.11,"20 mm line · 2 mm floor sampling\nOpen start-to-finish courses · arrows show direction\nLayouts only: no NEAT or PID tests have been run.",fontsize=10,color="#546778",va="top")
    fig.suptitle("Five fresh long-course candidates — performance-blind selection",fontsize=19)
    for suffix in ("png","svg"):
        fig.savefig(destination/f"five-course-overview.{suffix}",dpi=240,bbox_inches="tight")
    plt.close(fig)
    for recipe,track,blocks,record in courses:
        fig=plt.figure(figsize=(13,7.6),layout="constrained")
        grid=fig.add_gridspec(2,2,width_ratios=(3.1,1.4),height_ratios=(5.,1.))
        axis=fig.add_subplot(grid[0,0])
        selected=draw(track,blocks,axis,details=True)
        notes=fig.add_subplot(grid[0,1]); notes.axis("off")
        notes.text(0,1,"Key sections",fontsize=12,weight="bold",va="top")
        spacing=min(.083,.68/max(len(selected),1))
        for i,block in enumerate(selected,1):
            notes.text(0,.94-(i-1)*spacing,f"{i:>2}  {block['name']}\n      {block['length_m']:.2f} m",fontsize=8.4,va="top",linespacing=1.25)
        notes.text(0,.04,f"Route: {track.route.length:.2f} m\nFloor: {track.arena_size:.0f} × {track.arena_size:.0f} m\nSharp turns: {record['sharp_turns']}\nLine gaps: {record['gaps']} · distractors: {record['distractors']}",fontsize=9,va="bottom",color="#546778")
        timeline=fig.add_subplot(grid[1,:])
        colors={"straight":"#adc1d1","wide curve":"#008c95","tight curve":"#e69f00"}
        distance=0.
        for section in track.sections:
            length=section.length if section.kind=="straight" else abs(section.radius*section.angle) if section.kind=="arc" else 0.
            if section.kind=="corner":
                timeline.vlines(distance,.04,.96,color="#c34135",linewidth=1.1)
            else:
                kind="straight" if section.kind=="straight" else "wide curve" if section.radius>=.8 else "tight curve"
                timeline.broken_barh([(distance,length)],(.25,.5),facecolors=colors[kind])
            distance+=length
        for a,b in track.gaps:
            timeline.axvline((a+b)/2,color="#c43886",linewidth=2,linestyle="--")
        timeline.set(xlim=(0,track.route.length),ylim=(0,1),yticks=[],xlabel="Distance from start (m) — red ticks mark sharp turns; magenta ticks mark gaps")
        timeline.spines[["top","left","right"]].set_visible(False)
        handles=[Line2D([],[],color=color,lw=6,label=name) for name,color in colors.items()]
        timeline.legend(handles=handles,ncol=3,loc="upper center",bbox_to_anchor=(.5,1.35),fontsize=8,frameon=False)
        fig.suptitle(f"{recipe['id']}. {recipe['title']} — {track.route.length:.2f} m",fontsize=17)
        fig.text(.01,-.02,"20 mm line; 2 mm floor sampling. Orange map dots = sharp turns; purple diamonds = junctions; magenta rings = gaps. No controllers evaluated.",fontsize=9,color="#546778")
        for suffix in ("png","svg"):
            fig.savefig(destination/f"{recipe['id']:02d}-{recipe['slug']}.{suffix}",dpi=240,bbox_inches="tight")
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--check-only",action="store_true",help="Check layouts without writing files, rasterizing, or evaluating controllers")
    args=parser.parse_args()
    prior=protected_tracks()
    existing_suite=args.output.resolve()/"suite.json"
    if existing_suite.exists():
        existing=json.loads(existing_suite.read_text(encoding="utf-8"))
        prior.difference_update(e["geometry_hash"] for e in existing["splits"]["candidate"])
    previous=ROOT/"runs/long-course-speed-challenge/track.json"
    if previous.exists(): prior.add(TrackSpec.load(previous).geometry_hash)
    courses=[]; hashes=set()
    for recipe in recipes():
        track,blocks=build(recipe)
        clearance=check_geometry(track)
        if track.geometry_hash in prior or track.geometry_hash in hashes:
            raise ValueError("Candidate geometry repeats an existing or mirrored course")
        hashes.add(track.geometry_hash)
        record={"id":recipe["id"],"name":recipe["title"],"file":f"{recipe['id']:02d}-{recipe['slug']}.json",
                "focus":recipe["focus"],"description":recipe["description"],"length_m":track.route.length,
                "arena_side_m":track.arena_size,"sharp_turns":sum(s.kind=="corner" for s in track.sections),
                "arc_sections":sum(s.kind=="arc" for s in track.sections),"gaps":len(track.gaps),"distractors":len(track.branches),
                "minimum_nonlocal_route_clearance_m":clearance,"geometry_sha256":track.geometry_hash,"blocks":blocks}
        if not args.check_only: record.update(check_floor(track))
        courses.append((recipe,track,blocks,record))
        print(f"{recipe['id']}: {recipe['title']} | {track.route.length:.2f} m | {track.arena_size:.0f} m arena | {record['sharp_turns']} sharp turns | clearance {clearance:.3f} m",flush=True)
    if args.check_only: return
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    for recipe,track,blocks,record in courses:
        path=output/record["file"]
        if path.exists() and TrackSpec.load(path)!=track:
            raise ValueError("Preserve published candidates; choose a new output directory")
        track.save(path)
        if TrackSpec.load(path)!=track: raise ValueError("Track JSON roundtrip changed geometry")
        record["file_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
    figures(courses,output)
    manifest={"status":"Awaiting user course choice; no controller evaluation or evolution performed",
              "chosen_candidate":None,"controller_evaluations":0,
              "date":"2026-11-21","line_width_m":.020,"floor_resolution_m":.002,"recommended_candidate":1,
              "proposed_primary_cap_m_s":1.,"proposed_common_time_limit_s":180,
              "selection_note":"Choose from layouts and scientific purpose, before any controller outcomes. Freeze policies and final episode rules after the course choice, before testing.",
              "geometry_checks":"Bounds, radius, nonlocal clearance, runout, gap/junction placement, raster paint and gap checks, canonical separation from protected geometry including mirrors, and JSON roundtrip.",
              "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "candidates":[record for _,_,_,record in courses]}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    # Register these reserved geometries with the existing protected-suite scan.
    suite={"suite_version":1,"generator_version":"2.0","root_seed":2026112120,
           "purpose":"Reserved long-course choices; no controller outcomes observed",
           "splits":{"candidate":[{"path":r["file"],"geometry_hash":r["geometry_sha256"],
                                      "difficulty":"advanced","features":track.features}
                                     for _,track,_,r in courses]}}
    (output/"suite.json").write_text(json.dumps(suite,indent=2),encoding="utf-8")
    print(f"Saved five track JSON files and PNG/SVG previews to {output}. No controller has been run.")


if __name__=="__main__":
    main()
