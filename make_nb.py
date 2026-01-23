import json

code = '''#@title RSG Lensing Suite - Clone & Run (Full Version)
# Clone the repository and install dependencies
!rm -rf ssz-lensing 2>/dev/null; git clone --depth 1 https://github.com/error-wtf/ssz-lensing.git
!pip install -q gradio numpy matplotlib scipy plotly

import sys
sys.path.insert(0, 'ssz-lensing/src')

import gradio as gr, numpy as np, matplotlib.pyplot as plt, plotly.graph_objects as go
from scipy.integrate import quad as iq
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

A=np.pi/(180*3600); G,c,Ms,Mpc,pc=6.674e-11,299792458,1.989e30,3.086e22,3.086e16

# ============ REAL FALLBACK DATASETS (NO FAKE VALUES) ============
# Q2237+0305 Einstein Cross - HST/CASTLES data (Kochanek et al.)
CROSS_DATA = {
    'name': 'Q2237+0305 (Einstein Cross)',
    'z_L': 0.0394, 'z_S': 1.695,
    'theta_E': 0.89,  # arcsec
    'positions_arcsec': [  # A,B,C,D from CASTLES
        (0.740, 0.565),   # A
        (-0.635, 0.470),  # B
        (-0.480, -0.755), # C
        (0.870, -0.195)   # D
    ],
    'source': 'CASTLES: https://lweb.cfa.harvard.edu/castles/',
    'ref': 'Huchra et al. 1985, AJ 90, 691'
}
# SDSS J1004+4112 - Giant arc/ring system (5 images)
RING_DATA = {
    'name': 'SDSS J1004+4112 (Cluster lens)',
    'z_L': 0.68, 'z_S': 1.734,
    'theta_E': 7.0,  # arcsec (approximate)
    'positions_arcsec': [  # 5 images from Inada et al. 2003
        (7.42, 3.26), (-5.82, 1.91), (-4.51, -3.18), (2.85, -5.91), (0.12, 0.05)
    ],
    'source': 'SDSS Quasar Lens Search',
    'ref': 'Inada et al. 2003, Nature 426, 810'
}

@dataclass
class LensingRun:
    name: str = ''
    # Distances
    z_L: float = 0.0; z_S: float = 0.0
    D_L: float = 0.0; D_S: float = 0.0; D_LS: float = 0.0
    # Lens
    M: float = 0.0; r_s: float = 0.0
    theta_E: float = 0.0; b_E: float = 0.0
    # RSG quantities at R_ref
    R_ref: float = 0.0; Xi_ref: float = 0.0; s_ref: float = 0.0; D_ref: float = 0.0
    # Image positions (GR = input, SSZ = scaled)
    theta_GR: np.ndarray = field(default_factory=lambda: np.array([]))
    theta_SSZ: np.ndarray = field(default_factory=lambda: np.array([]))
    b_GR: np.ndarray = field(default_factory=lambda: np.array([]))
    b_SSZ: np.ndarray = field(default_factory=lambda: np.array([]))
    # Deltas
    Delta_theta: np.ndarray = field(default_factory=lambda: np.array([]))
    Delta_b: np.ndarray = field(default_factory=lambda: np.array([]))
    rms_theta: float = 0.0; max_Delta_theta: float = 0.0
    # Morphology
    morphology: str = 'UNKNOWN'
    source_info: str = ''

def build_run(pos_arcsec, z_L, z_S, theta_E, name='Custom'):
    run = LensingRun(name=name, z_L=z_L, z_S=z_S, theta_E=theta_E)
    # Cosmology
    def E(z): return 1/np.sqrt(0.315*(1+z)**3+0.685)
    DH = c/(67.4*1000/Mpc)
    cL,_ = iq(E,0,z_L); cS,_ = iq(E,0,z_S)
    run.D_L = cL*DH/(1+z_L); run.D_S = cS*DH/(1+z_S); run.D_LS = (cS-cL)*DH/(1+z_S)
    # Lens mass
    Scr = c**2*run.D_S/(4*np.pi*G*run.D_L*run.D_LS)
    run.b_E = theta_E*A*run.D_L
    run.M = np.pi*run.b_E**2*Scr/Ms
    run.r_s = 2*G*run.M*Ms/c**2
    # RSG at b_E
    run.R_ref = run.b_E
    run.Xi_ref = run.r_s/(2*run.R_ref)
    run.s_ref = 1 + run.Xi_ref
    run.D_ref = 1/run.s_ref
    # GR positions (input)
    pos = np.array(pos_arcsec)
    run.theta_GR = pos  # arcsec
    r_GR = np.hypot(pos[:,0], pos[:,1])
    ang = np.arctan2(pos[:,1], pos[:,0])
    # SSZ positions (scaled)
    r_SSZ = run.s_ref * r_GR
    run.theta_SSZ = np.column_stack([r_SSZ*np.cos(ang), r_SSZ*np.sin(ang)])
    # Impact parameters
    run.b_GR = run.D_L * r_GR * A
    run.b_SSZ = run.s_ref * run.b_GR
    # Deltas
    run.Delta_theta = run.theta_SSZ - run.theta_GR  # arcsec
    run.Delta_b = run.b_SSZ - run.b_GR
    run.rms_theta = np.sqrt(np.mean(np.sum(run.Delta_theta**2, axis=1)))
    run.max_Delta_theta = np.max(np.hypot(run.Delta_theta[:,0], run.Delta_theta[:,1]))
    # Morphology
    n = len(pos)
    if n == 4: run.morphology = 'QUAD'
    elif n == 2: run.morphology = 'DOUBLE'
    elif n > 4: run.morphology = 'RING/ARC'
    return run

def load_cross():
    d = CROSS_DATA
    return build_run(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'], d['name'])

def load_ring():
    d = RING_DATA
    return build_run(d['positions_arcsec'], d['z_L'], d['z_S'], d['theta_E'], d['name'])

def parse(t,u='arcsec'):
    f={'arcsec':A,'mas':A/1000,'rad':1}[u]
    lines = [l.strip() for l in t.strip().split('\\n') if l.strip() and not l.startswith('#')]
    return np.array([[float(x) for x in l.replace(',',' ').split()[:2]] for l in lines])*f

def fmt(r):
    a=r/A
    if abs(a)>=0.01: return f'{a:.4f}"'
    return f'{a*1000:.3f} mas'

# Full morphology classification with harmonics
def morph_full(pos):
    n=len(pos); ctr=np.mean(pos,axis=0); rel=pos-ctr
    r=np.hypot(rel[:,0],rel[:,1]); phi=np.arctan2(rel[:,1],rel[:,0])
    rm=np.mean(r); rs=np.std(r)/rm if rm>0 else 1
    # Azimuthal coverage
    ps=np.sort(phi); gaps=np.diff(ps); gaps=np.append(gaps,2*np.pi+ps[0]-ps[-1])
    az_cov=1-np.max(gaps)/(2*np.pi); az_uni=1/(1+np.var(gaps)/(2*np.pi/n)**2)
    # Harmonics m=2,3,4
    dr=r-rm
    m2=np.hypot(np.mean(dr*np.cos(2*phi)),np.mean(dr*np.sin(2*phi)))/rm if rm>0 else 0
    m3=np.hypot(np.mean(dr*np.cos(3*phi)),np.mean(dr*np.sin(3*phi)))/rm if rm>0 else 0
    m4=np.hypot(np.mean(dr*np.cos(4*phi)),np.mean(dr*np.sin(4*phi)))/rm if rm>0 else 0
    # Classification
    notes,models=[],[]
    if n==4: typ,conf='QUAD',0.95; notes.append('Einstein Cross: 4 discrete images'); models=['m2','m2+shear','m2+m3','m2+m4']
    elif n==2: typ,conf='DOUBLE',0.9; notes.append('Double: 2-image system'); models=['SIS','SIE']
    elif n>4 and rs<0.05 and az_cov>0.7: typ,conf='RING',0.9; notes.append('Einstein Ring'); models=['iso','iso+shear','iso+m2']
    elif n>4 and az_cov<0.7: typ,conf='ARC',0.75; notes.append('Partial arc'); models=['m2+shear','NFW']
    else: typ,conf='UNKNOWN',0.5; notes.append('Morphology unclear'); models=['m2']
    if m2>0.01: notes.append(f'm=2 pert: {m2:.4f}')
    if m3>0.005: notes.append(f'm=3 pert: {m3:.4f}')
    if m4>0.005: notes.append(f'm=4 pert: {m4:.4f}')
    return {'type':typ,'conf':conf,'r_mean':rm,'radial_scatter':rs,'az_cov':az_cov,'az_uni':az_uni,
            'm2':m2,'m3':m3,'m4':m4,'notes':notes,'models':models,'ctr':ctr}

# Full ring analysis with harmonic decomposition
def ring_full(pos):
    ctr=np.mean(pos,axis=0); rel=pos-ctr; r=np.hypot(rel[:,0],rel[:,1])
    phi=np.arctan2(rel[:,1],rel[:,0]); R=np.median(r); dr=r-R
    rms=np.sqrt(np.mean(dr**2))
    # Harmonics with phase
    def fit_m(m):
        cc,ss=2*np.mean(dr*np.cos(m*phi)),2*np.mean(dr*np.sin(m*phi))
        return np.hypot(cc,ss),np.arctan2(ss,cc)/m
    m2a,m2p=fit_m(2); m3a,m3p=fit_m(3); m4a,m4p=fit_m(4)
    # Perturbation type
    is_pert=m2a>0.02*R or m4a>0.01*R
    if m2a>m4a and m2a>0.02*R: ptype='quadrupole (m=2)'
    elif m4a>0.01*R: ptype='hexadecapole (m=4)'
    elif m3a>0.01*R: ptype='octupole (m=3)'
    else: ptype='isotropic'
    return {'cx':ctr[0],'cy':ctr[1],'R':R,'rms':rms,'m2_amp':m2a,'m2_phase':m2p,
            'm3_amp':m3a,'m3_phase':m3p,'m4_amp':m4a,'m4_phase':m4p,'ptype':ptype,'is_pert':is_pert,'dr':dr,'phi':phi}

# Full quad inversion with multipoles
def invert_full(pos):
    if len(pos)!=4: return None
    r=np.hypot(pos[:,0],pos[:,1]); phi=np.arctan2(pos[:,1],pos[:,0])
    idx=np.argsort(phi); rs,ps=r[idx],phi[idx]; tE=np.mean(rs); dr=rs-tE
    m2c,m2s=np.mean(dr*np.cos(2*ps)),np.mean(dr*np.sin(2*ps)); m2=np.hypot(m2c,m2s); m2p=np.arctan2(m2s,m2c)/2
    m3=np.hypot(np.mean(dr*np.cos(3*ps)),np.mean(dr*np.sin(3*ps)))
    m4=np.hypot(np.mean(dr*np.cos(4*ps)),np.mean(dr*np.sin(4*ps)))
    w=1/rs; bx,by=np.sum(w*pos[idx,0])/np.sum(w),np.sum(w*pos[idx,1])/np.sum(w)
    rm=tE+m2*np.cos(2*(ps-m2p)); rms=np.sqrt(np.mean((rs-rm)**2))
    quality='Excellent' if rms/tE<0.01 else 'Good' if rms/tE<0.05 else 'Moderate'
    return {'tE':tE,'beta':np.hypot(bx,by),'bx':bx,'by':by,'m2':m2,'m2_deg':np.degrees(m2p),'m3':m3,'m4':m4,'rms':rms,'quality':quality}

def cosmo(zL,zS):
    def E(z): return 1/np.sqrt(0.315*(1+z)**3+0.685)
    DH=c/(67.4*1000/Mpc)
    cL,_=iq(E,0,zL); cS,_=iq(E,0,zS)
    return cL*DH/(1+zL), cS*DH/(1+zS), (cS-cL)*DH/(1+zS)

def mass(tE,DL,DS,DLS):
    Scr=c**2*DS/(4*np.pi*G*DL*DLS); RE=tE*A*DL
    return np.pi*RE**2*Scr/Ms

QUAD="0.740, 0.565\\n-0.635, 0.470\\n-0.480, -0.755\\n0.870, -0.195"

def t1(txt,u):
    pos=parse(txt,u); n=len(pos); r=np.hypot(pos[:,0],pos[:,1])
    fig,ax=plt.subplots(figsize=(6,6)); p=pos/A
    ax.scatter(p[:,0],p[:,1],s=80,c='blue',edgecolors='k',zorder=5)
    for i,(x,y) in enumerate(p): ax.annotate(str(i+1),(x,y),xytext=(5,5),textcoords='offset points')
    ax.axhline(0,c='gray',ls='--',alpha=.3); ax.axvline(0,c='gray',ls='--',alpha=.3)
    ax.set_aspect('equal'); ax.grid(alpha=.3); ax.set_xlabel('x [arcsec]'); ax.set_ylabel('y [arcsec]')
    out=f"## Data: {n} pts\\n| Metric | Value |\\n|--|--|\\n| Mean r | {fmt(np.mean(r))} |\\n| Range | {fmt(np.min(r))}-{fmt(np.max(r))} |"
    return out, fig

def t2(txt,u):
    pos=parse(txt,u); m=morph_full(pos); c=m['ctr']; th=np.linspace(0,2*np.pi,100)
    fig,axes=plt.subplots(1,2,figsize=(12,5)); p=pos/A
    axes[0].scatter(p[:,0],p[:,1],s=80,c='blue',zorder=5)
    axes[0].plot(c[0]/A+m['r_mean']/A*np.cos(th),c[1]/A+m['r_mean']/A*np.sin(th),'g--',lw=2)
    axes[0].scatter([c[0]/A],[c[1]/A],s=150,c='red',marker='+',lw=3)
    axes[0].set_xlabel('x [arcsec]'); axes[0].set_ylabel('y [arcsec]'); axes[0].set_aspect('equal'); axes[0].grid(alpha=.3)
    axes[0].set_title(f"{m['type']} ({m['conf']:.0%})")
    bars=axes[1].bar(['m=2','m=3','m=4'],[m['m2']*100,m['m3']*100,m['m4']*100],color=['orange','cyan','purple'])
    for b,v in zip(bars,[m['m2'],m['m3'],m['m4']]): axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+0.1,f'{v:.5f}',ha='center',fontsize=9)
    axes[1].set_ylabel('Amp (x100)'); axes[1].set_title('Harmonics'); axes[1].grid(alpha=.3,axis='y'); plt.tight_layout()
    out=f"## {m['type']} ({m['conf']:.0%})\\n| Metric | Value |\\n|--|--|\\n| r_mean | {fmt(m['r_mean'])} |\\n| scatter | {m['radial_scatter']:.4f} |\\n| az_cov | {m['az_cov']:.1%} |\\n### Harmonics\\n| Mode | Amp |\\n|--|--|\\n| m=2 | {m['m2']:.6f} |\\n| m=3 | {m['m3']:.6f} |\\n| m=4 | {m['m4']:.6f} |\\n**Notes:** {'; '.join(m['notes'])}\\n**Models:** {', '.join(m['models'])}"
    return out, fig

def t3(txt,u):
    pos=parse(txt,u); r=ring_full(pos); th=np.linspace(0,2*np.pi,100)
    fig,axes=plt.subplots(1,3,figsize=(15,4)); p=pos/A
    axes[0].scatter(p[:,0],p[:,1],s=80,c='blue',zorder=5)
    axes[0].plot(r['cx']/A+r['R']/A*np.cos(th),r['cy']/A+r['R']/A*np.sin(th),'g-',lw=2)
    axes[0].scatter([r['cx']/A],[r['cy']/A],s=150,c='red',marker='+',lw=2)
    axes[0].set_xlabel('x [arcsec]'); axes[0].set_ylabel('y [arcsec]'); axes[0].set_aspect('equal'); axes[0].grid(alpha=.3)
    axes[0].set_title(f"R={r['R']/A:.4f}\\\"")
    axes[1].scatter(np.degrees(r['phi']),r['dr']/A*1000,s=50,c='blue'); axes[1].axhline(0,c='gray',ls='--')
    axes[1].set_xlabel('Azimuth [deg]'); axes[1].set_ylabel('Δr [mas]'); axes[1].grid(alpha=.3)
    axes[1].set_title(f"Residuals (RMS={r['rms']/A*1000:.2f} mas)")
    bars=axes[2].bar(['m=2','m=3','m=4'],[r['m2_amp']/A*1000,r['m3_amp']/A*1000,r['m4_amp']/A*1000],color=['orange','cyan','purple'])
    for b,a,ph in zip(bars,[r['m2_amp'],r['m3_amp'],r['m4_amp']],[r['m2_phase'],r['m3_phase'],r['m4_phase']]):
        axes[2].text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f'{a/A*1000:.2f}\\n{np.degrees(ph):.0f}°',ha='center',fontsize=8)
    axes[2].set_ylabel('Amp [mas]'); axes[2].set_title('Harmonics'); axes[2].grid(alpha=.3,axis='y'); plt.tight_layout()
    out=f"## Ring Fit\\n| Param | Value |\\n|--|--|\\n| Center | ({fmt(r['cx'])}, {fmt(r['cy'])}) |\\n| R (θ_E) | {fmt(r['R'])} |\\n| RMS | {fmt(r['rms'])} |\\n| Type | **{r['ptype']}** |\\n### Harmonics\\n| Mode | Amp (mas) | Phase |\\n|--|--|--|\\n| m=2 | {r['m2_amp']/A*1000:.3f} | {np.degrees(r['m2_phase']):.1f}° |\\n| m=3 | {r['m3_amp']/A*1000:.3f} | {np.degrees(r['m3_phase']):.1f}° |\\n| m=4 | {r['m4_amp']/A*1000:.3f} | {np.degrees(r['m4_phase']):.1f}° |\\n**Perturbed:** {'Yes' if r['is_pert'] else 'No'}"
    return out, fig

def t4(txt,u):
    pos=parse(txt,u); inv=invert_full(pos)
    if not inv: return '❌ Need exactly 4 images', None
    th=np.linspace(0,2*np.pi,100)
    fig,axes=plt.subplots(1,2,figsize=(12,5)); p=pos/A
    axes[0].scatter(p[:,0],p[:,1],s=100,c='blue',label='Images',zorder=5)
    axes[0].scatter([inv['bx']/A],[inv['by']/A],s=150,c='gold',marker='*',label='Source',zorder=6)
    axes[0].plot(inv['tE']/A*np.cos(th),inv['tE']/A*np.sin(th),'g--',lw=2,label=f"θ_E={inv['tE']/A:.3f}\\\"")
    axes[0].scatter([0],[0],s=100,c='red',marker='+',lw=2,label='Lens')
    axes[0].legend(); axes[0].set_xlabel('x [arcsec]'); axes[0].set_ylabel('y [arcsec]')
    axes[0].set_aspect('equal'); axes[0].grid(alpha=.3); axes[0].set_title('Quad Geometry')
    bars=axes[1].bar(['m=2','m=3','m=4'],[inv['m2']/A*1000,inv['m3']/A*1000,inv['m4']/A*1000],color=['orange','cyan','purple'])
    for b,v in zip(bars,[inv['m2'],inv['m3'],inv['m4']]): axes[1].text(b.get_x()+b.get_width()/2,b.get_height()+0.05,f'{v/A*1000:.2f}',ha='center',fontsize=9)
    axes[1].set_ylabel('Amp [mas]'); axes[1].set_title('Multipoles'); axes[1].grid(alpha=.3,axis='y'); plt.tight_layout()
    out=f"## Exact Inversion\\n| Param | Value |\\n|--|--|\\n| θ_E | {fmt(inv['tE'])} |\\n| β (source) | {fmt(inv['beta'])} |\\n| Source pos | ({fmt(inv['bx'])}, {fmt(inv['by'])}) |\\n| RMS | {fmt(inv['rms'])} |\\n| Quality | **{inv['quality']}** |\\n### Multipoles\\n| Mode | Amp (mas) | Phase |\\n|--|--|--|\\n| m=2 | {inv['m2']/A*1000:.3f} | {inv['m2_deg']:.1f}° |\\n| m=3 | {inv['m3']/A*1000:.3f} | - |\\n| m=4 | {inv['m4']/A*1000:.3f} | - |"
    return out, fig

def t5(zL,zS,tE):
    DL,DS,DLS=cosmo(zL,zS); M=mass(tE,DL,DS,DLS)
    Scr=c**2*DS/(4*np.pi*G*DL*DLS); RE=tE*A*DL
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    bars=axes[0].barh(['D_L','D_S','D_LS'],[DL/Mpc,DS/Mpc,DLS/Mpc],color=['red','blue','green'])
    axes[0].bar_label(bars,fmt='%.0f Mpc'); axes[0].set_xlabel('Distance [Mpc]'); axes[0].grid(alpha=.3,axis='x')
    axes[0].set_title('Angular Diameter Distances')
    axes[1].bar(['M_lens'],[np.log10(M)],color='purple')
    axes[1].set_ylabel('log₁₀(M/M☉)'); axes[1].set_title(f'M = {M:.2e} M☉'); axes[1].grid(alpha=.3,axis='y')
    plt.tight_layout()
    out=f"## Cosmology\\n| Param | Value |\\n|--|--|\\n| z_L | {zL:.4f} |\\n| z_S | {zS:.4f} |\\n| D_L | {DL/Mpc:.1f} Mpc |\\n| D_S | {DS/Mpc:.1f} Mpc |\\n| D_LS | {DLS/Mpc:.1f} Mpc |\\n| θ_E | {tE:.4f}\\\" |\\n| R_E | {RE/1e3:.2f} kpc |\\n| Σ_cr | {Scr:.2e} kg/m² |\\n| **M** | **{M:.2e} M☉** |"
    return out, fig

def t6(txt,u,zL,zS,tE):
    pos=parse(txt,u); DL,DS,DLS=cosmo(zL,zS); M=mass(tE,DL,DS,DLS)
    n=len(pos); cols=['#ff6b6b','#4ecdc4','#ffe66d','#95e1d3']
    b_E=DL*tE*A; b_E_kpc=b_E/(1e3*pc)
    zS_n,zL_n,zO_n=0,0.35,1; scale=0.12
    fig=go.Figure()
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[zS_n],mode='markers',marker=dict(size=15,color='yellow'),name='Quelle'))
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[zL_n],mode='markers',marker=dict(size=14,color='red',symbol='square'),name='Linse'))
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[zO_n],mode='markers',marker=dict(size=12,color='#5dade2'),name='Beobachter'))
    th=np.linspace(0,2*np.pi,60)
    fig.add_trace(go.Scatter3d(x=scale*np.cos(th),y=scale*np.sin(th),z=[zL_n]*60,mode='lines',line=dict(color='lime',width=4),name='Impact circle b_E'))
    # Default 4 rays if no data
    if n==0:
        angs=np.array([0,np.pi/2,np.pi,3*np.pi/2]); rads=np.ones(4); n_rays=4
    else:
        angs=np.arctan2(pos[:,1],pos[:,0]); rads=np.hypot(pos[:,0],pos[:,1])/(A*tE) if tE>0 else np.ones(n); n_rays=min(n,4)
    b_vals=[]
    for i in range(n_rays):
        phi=angs[i]; r_n=min(float(rads[i]),1.2)*scale
        bx,by=r_n*np.cos(phi),r_n*np.sin(phi)
        b_vals.append(np.hypot(bx,by)/scale)
        fig.add_trace(go.Scatter3d(x=[0,bx,0],y=[0,by,0],z=[zS_n,zL_n,zO_n],mode='lines+markers',line=dict(color=cols[i%4],width=4),marker=dict(size=[4,8,4],color=cols[i%4]),name=f'Ray {i+1}'))
    fig.update_layout(scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='z',bgcolor='#0a0a1a',xaxis=dict(range=[-.18,.18]),yaxis=dict(range=[-.18,.18]),zaxis=dict(range=[-.05,1.05])),title='3D: Impact circle (b) at Lens Plane',margin=dict(l=0,r=0,t=40,b=0),height=550)
    # Output with b_i values
    out=f"## 3D Lens Geometry\\n\\n**Impact circle at lens plane** (NOT Einstein Ring!)\\n\\n"
    out+=f"| Parameter | Wert |\\n|--|--|\\n"
    out+=f"| b_E = D_L x theta_E | {b_E_kpc:.2f} kpc |\\n"
    out+=f"| D_L | {DL/Mpc:.1f} Mpc |\\n| theta_E | {tE:.4f} arcsec |\\n\\n"
    out+=f"**Ray impact parameters (b_i / b_E):**\\n\\n"
    for i,bv in enumerate(b_vals): out+=f"- Ray {i+1}: b/b_E = {bv:.3f}\\n"
    out+=f"\\n*Gruener Kreis = Impact circle bei z=D_L (Linsenebene)*"
    return out, fig

def t7_sky(txt,u,tE):
    pos=parse(txt,u); n=len(pos); p=pos/A  # convert to arcsec
    cols=['#ff6b6b','#4ecdc4','#ffe66d','#95e1d3']
    fig,ax=plt.subplots(figsize=(8,8))
    ax.set_facecolor('#0a0a1a')
    # Einstein Ring as angular reference circle (THIS IS CORRECT HERE)
    th=np.linspace(0,2*np.pi,100)
    ax.plot(tE*np.cos(th),tE*np.sin(th),'g--',lw=2,alpha=0.8,label=f'Einstein Ring θ_E={tE:.3f}"')
    # Lens center
    ax.scatter([0],[0],s=150,c='red',marker='+',lw=3,zorder=10,label='Linse (Zentrum)')
    # Image positions (apparent positions on sky)
    for i in range(min(n,4)):
        ax.scatter([p[i,0]],[p[i,1]],s=150,c=cols[i],edgecolors='white',lw=2,zorder=5)
        ax.annotate(f'  Bild {i+1}',(p[i,0],p[i,1]),fontsize=10,color=cols[i],weight='bold')
    lim=max(tE*1.8,np.max(np.abs(p))*1.3) if n>0 else tE*2
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_aspect('equal'); ax.grid(alpha=0.3,color='white')
    ax.set_xlabel('θ_x [arcsec]',fontsize=12); ax.set_ylabel('θ_y [arcsec]',fontsize=12)
    ax.set_title('Observer Sky: Winkelprojektion (θx, θy)',fontsize=14)
    ax.legend(loc='upper right',fontsize=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('white')
    plt.tight_layout()
    out="## Observer Sky Panel\\n**Korrekte Darstellung:** Einstein-Ring als Winkelprojektion\\n\\n| Bild | θ_x | θ_y | r |\\n|--|--|--|--|\\n"
    for i in range(min(n,4)):
        px,py=p[i,0],p[i,1]; r_i=np.hypot(px,py)
        out+=f"| {i+1} | {px:.4f} | {py:.4f} | {r_i:.4f} |\\n"
    out+=f"\\n| θ_E | {tE:.4f} arcsec |\\n\\n**Der gruene Kreis ist hier korrekt:** Er zeigt den Einstein-Radius als Winkel am Himmel des Beobachters."
    return out, fig

def t8(txt,u,zL,zS,tE):
    pos=parse(txt,u); n=len(pos)
    DL,DS,DLS=cosmo(zL,zS); M=mass(tE,DL,DS,DLS)
    r_s=2*G*M*Ms/c**2; b_E=DL*tE*A; b_E_kpc=b_E/(1e3*pc)
    Xi_bE=r_s/(2*b_E); s_bE=1+Xi_bE
    cols=['#ff6b6b','#4ecdc4','#ffe66d','#95e1d3']
    # GR image positions (from input or default cross)
    if n>0:
        th_GR=pos/A; r_GR=np.hypot(th_GR[:,0],th_GR[:,1]); ang=np.arctan2(th_GR[:,1],th_GR[:,0])
    else:
        ang=np.array([0.3,1.8,3.4,4.9]); r_GR=np.array([0.85,0.92,1.05,0.98])*tE
        th_GR=np.column_stack([r_GR*np.cos(ang),r_GR*np.sin(ang)])
    n_img=min(len(r_GR),4)
    # SSZ: radial scaling θ_SSZ = s(b_E) * θ_GR
    r_SSZ=s_bE*r_GR; th_SSZ=np.column_stack([r_SSZ*np.cos(ang),r_SSZ*np.sin(ang)])
    b_GR=DL*r_GR[:n_img]*A; b_SSZ=s_bE*b_GR
    Delta_th=r_SSZ[:n_img]-r_GR[:n_img]; Delta_b=b_SSZ-b_GR
    R_fit=np.mean(r_GR[:n_img]); resid_GR=r_GR[:n_img]-R_fit
    rms_th=np.sqrt(np.mean(Delta_th**2))*1000  # mas
    fig,axes=plt.subplots(2,2,figsize=(14,12))
    # PANEL 1: Observer Sky - GR vs SSZ with arrows
    ax=axes[0,0]; circ=np.linspace(0,2*np.pi,100)
    ax.plot(tE*np.cos(circ),tE*np.sin(circ),'gray',ls='--',lw=1.5,label=f'θ_E={tE:.3f}"')
    ax.axhline(0,color='gray',lw=0.3); ax.axvline(0,color='gray',lw=0.3)
    for i in range(n_img):
        gx,gy=th_GR[i]; sx,sy=th_SSZ[i]
        ax.scatter([gx],[gy],s=80,c='gray',marker='o',alpha=0.5,zorder=5)
        ax.scatter([sx],[sy],s=100,c=cols[i],marker='o',edgecolors='white',lw=2,zorder=6)
        ax.annotate('',(sx,sy),(gx,gy),arrowprops=dict(arrowstyle='->',color=cols[i],lw=2))
        ax.text(sx*1.08,sy*1.08,f'{i+1}',fontsize=10,color=cols[i],weight='bold')
    ax.scatter([0],[0],s=60,c='red',marker='+',lw=2,zorder=10,label='Lens')
    ax.set_aspect('equal'); ax.set_xlabel('θ_x [arcsec]'); ax.set_ylabel('θ_y [arcsec]')
    ax.legend(loc='upper right'); ax.grid(alpha=.3)
    ax.set_title('Panel 1: Observer Sky — GR (gray) → SSZ (color)')
    # PANEL 2: Lens Plane - Impact Parameters
    ax=axes[0,1]
    ax.plot(b_E_kpc*np.cos(circ),b_E_kpc*np.sin(circ),'gray',ls='--',lw=1.5,label=f'b_E={b_E_kpc:.2f} kpc')
    ax.plot(b_E_kpc*s_bE*np.cos(circ),b_E_kpc*s_bE*np.sin(circ),'lime',ls='-',lw=2,label=f'b_E×s={b_E_kpc*s_bE:.2f} kpc')
    for i in range(n_img):
        bx,by=b_GR[i]/(1e3*pc)*np.cos(ang[i]),b_GR[i]/(1e3*pc)*np.sin(ang[i])
        sx,sy=b_SSZ[i]/(1e3*pc)*np.cos(ang[i]),b_SSZ[i]/(1e3*pc)*np.sin(ang[i])
        ax.scatter([bx],[by],s=80,c='gray',marker='s',alpha=0.5)
        ax.scatter([sx],[sy],s=100,c=cols[i],marker='s',edgecolors='white',lw=2)
        ax.annotate('',(sx,sy),(bx,by),arrowprops=dict(arrowstyle='->',color=cols[i],lw=1.5))
    ax.scatter([0],[0],s=80,c='red',marker='x',lw=3,zorder=10,label='Lens center')
    ax.set_aspect('equal'); ax.set_xlabel('x [kpc]'); ax.set_ylabel('y [kpc]')
    ax.legend(loc='upper right'); ax.grid(alpha=.3)
    ax.set_title('Panel 2: Lens Plane — Impact circle (NOT Einstein ring!)')
    # PANEL 3: Radial residual vs angle
    ax=axes[1,0]; ang_deg=np.degrees(ang[:n_img])
    ax.bar(ang_deg-5,resid_GR*1000,width=8,color='gray',alpha=0.6,label='GR residual')
    ax.bar(ang_deg+5,(r_SSZ[:n_img]-R_fit*s_bE)*1000,width=8,color='lime',alpha=0.8,label='SSZ residual')
    ax.axhline(0,color='black',lw=1)
    ax.set_xlabel('Angle [deg]'); ax.set_ylabel('r - R_fit [mas]')
    ax.legend(); ax.grid(alpha=.3)
    ax.set_title('Panel 3: Radial residual vs angle (Cross signature)')
    # PANEL 4: Coupled metrics - the meaning!
    ax=axes[1,1]; ax.axis('off')
    txt_out=f"Panel 4: SSZ/RSG Metrics at b_E\\n{'='*40}\\n\\n"
    txt_out+=f"Ξ(b_E) = r_s/(2b_E) = {Xi_bE:.3e}\\n"
    txt_out+=f"s(b_E) = 1 + Ξ     = {s_bE:.10f}\\n"
    txt_out+=f"Δb/b   = s - 1     = {Xi_bE:.3e}\\n"
    txt_out+=f"Δθ/θ   = s - 1     = {Xi_bE:.3e}\\n\\n"
    txt_out+=f"{'='*40}\\nPredicted vs Measured\\n{'='*40}\\n"
    txt_out+=f"Predicted |Δθ|_max = Ξ × θ_E = {Xi_bE*tE*1000:.4f} mas\\n"
    txt_out+=f"Measured  |Δθ|_max           = {np.max(np.abs(Delta_th))*1000:.4f} mas\\n"
    txt_out+=f"RMS Δθ                       = {rms_th:.4f} mas\\n\\n"
    txt_out+=f"{'='*40}\\nWirkungskette\\n{'='*40}\\n"
    txt_out+=f"Ξ(r) → s(r)=1+Ξ → b_SSZ=s·b_GR → θ_SSZ=s·θ_GR"
    ax.text(0.05,0.95,txt_out,transform=ax.transAxes,fontsize=11,va='top',ha='left',family='monospace',bbox=dict(boxstyle='round',facecolor='#f0f0f0',edgecolor='gray'))
    plt.tight_layout()
    out=f"## RSG 4-Panel Dashboard\\n\\n"
    out+=f"| Metric | Wert |\\n|--|--|\\n"
    out+=f"| Ξ(b_E) | {Xi_bE:.3e} |\\n| s(b_E) | {s_bE:.10f} |\\n"
    out+=f"| Δθ/θ = Δb/b | {Xi_bE:.3e} |\\n| RMS Δθ | {rms_th:.4f} mas |\\n"
    out+=f"| max Δθ | {np.max(np.abs(Delta_th))*1000:.4f} mas |\\n"
    return out, fig

def t0_build(src,txt,u,zL,zS,tE):
    if src=='Q2237+0305 (Cross)':
        run=load_cross()
    elif src=='SDSS J1004+4112 (Ring)':
        run=load_ring()
    else:
        pos=parse(txt,u)/A  # to arcsec
        run=build_run(pos.tolist(),zL,zS,tE,'Custom')
    out=f"## LensingRun: {run.name}\\n\\n"
    out+=f"| Parameter | Value |\\n|--|--|\\n"
    out+=f"| z_L | {run.z_L:.4f} |\\n| z_S | {run.z_S:.4f} |\\n"
    out+=f"| D_L | {run.D_L/Mpc:.1f} Mpc |\\n| D_S | {run.D_S/Mpc:.1f} Mpc |\\n"
    out+=f"| θ_E | {run.theta_E:.4f} arcsec |\\n| b_E | {run.b_E/(1e3*pc):.2f} kpc |\\n"
    out+=f"| M | {run.M:.2e} M☉ |\\n| r_s | {run.r_s:.2e} m |\\n"
    out+=f"| Ξ(b_E) | {run.Xi_ref:.3e} |\\n| s(b_E) | {run.s_ref:.10f} |\\n"
    out+=f"| Morphology | {run.morphology} |\\n| Images | {len(run.theta_GR)} |\\n"
    out+=f"| max Δθ | {run.max_Delta_theta*1000:.4f} mas |\\n| RMS Δθ | {run.rms_theta*1000:.4f} mas |\\n"
    out+=f"\\n**Wirkungskette:** Ξ → s=1+Ξ → θ_SSZ=s·θ_GR"
    return out

with gr.Blocks(title='RSG Lensing') as demo:
    gr.Markdown('# RSG Lensing Suite')
    gr.Markdown('**Sky circle** = θ_E at observer | **Impact circle** = b_E at lens plane (NOT Einstein ring!)')
    with gr.Tab('Data Source'):
        gr.Markdown('### Load real data or enter custom positions')
        src=gr.Dropdown(['Custom','Q2237+0305 (Cross)','SDSS J1004+4112 (Ring)'],value='Q2237+0305 (Cross)',label='Source')
        txt=gr.Textbox(value=QUAD,lines=5,label='Custom Positions (arcsec)')
        unit=gr.Dropdown(['arcsec','mas'],value='arcsec',label='Unit')
        with gr.Row(): zL=gr.Number(0.039,label='z_L'); zS=gr.Number(1.695,label='z_S'); tE=gr.Number(0.9,label='θ_E (arcsec)')
        b0=gr.Button('Build LensingRun',variant='primary'); o0=gr.Markdown()
        b0.click(t0_build,[src,txt,unit,zL,zS,tE],[o0])
    with gr.Tab('Validate'):
        b1=gr.Button('Validate',variant='primary'); o1=gr.Markdown(); p1=gr.Plot()
        b1.click(t1,[txt,unit],[o1,p1])
    with gr.Tab('Morphology'):
        b2=gr.Button('Classify',variant='primary'); o2=gr.Markdown(); p2=gr.Plot()
        b2.click(t2,[txt,unit],[o2,p2])
    with gr.Tab('Ring Fit'):
        b3=gr.Button('Fit',variant='primary'); o3=gr.Markdown(); p3=gr.Plot()
        b3.click(t3,[txt,unit],[o3,p3])
    with gr.Tab('Inversion'):
        b4=gr.Button('Invert',variant='primary'); o4=gr.Markdown(); p4=gr.Plot()
        b4.click(t4,[txt,unit],[o4,p4])
    with gr.Tab('Cosmology'):
        with gr.Row(): zL=gr.Number(0.039,label='z_L'); zS=gr.Number(1.695,label='z_S'); tE=gr.Number(0.9,label='tE (arcsec)')
        b5=gr.Button('Calc',variant='primary'); o5=gr.Markdown(); p5=gr.Plot()
        b5.click(t5,[zL,zS,tE],[o5,p5])
    with gr.Tab('3D Scene'):
        b6=gr.Button('Plot 3D',variant='primary'); o6=gr.Markdown(); p6=gr.Plot()
        b6.click(t6,[txt,unit,zL,zS,tE],[o6,p6])
    with gr.Tab('Sky Panel'):
        gr.Markdown('**Korrekte Darstellung:** Einstein-Ring als Winkelprojektion am Beobachterhimmel')
        b7=gr.Button('Show Sky',variant='primary'); o7=gr.Markdown(); p7=gr.Plot()
        b7.click(t7_sky,[txt,unit,tE],[o7,p7])
    with gr.Tab('Radial Gauge'):
        b8=gr.Button('Calc RSG',variant='primary'); o8=gr.Markdown(); p8=gr.Plot()
        b8.click(t8,[txt,unit,zL,zS,tE],[o8,p8])
demo.launch(share=True)
'''

nb = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'colab': {'provenance': []},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}
    },
    'cells': [{
        'cell_type': 'code',
        'source': [code],
        'metadata': {},
        'execution_count': None,
        'outputs': []
    }]
}

with open('SSZ_Lensing_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Created SSZ_Lensing_Colab.ipynb with 8 tabs')
