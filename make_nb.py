import json

code = '''#@title RSG Lensing Suite - Clone & Run (Full Version)
# Clone the repository and install dependencies
!rm -rf ssz-lensing 2>/dev/null; git clone --depth 1 https://github.com/error-wtf/ssz-lensing.git
!pip install -q gradio numpy matplotlib scipy plotly

import sys
sys.path.insert(0, 'ssz-lensing/src')

import gradio as gr, numpy as np, matplotlib.pyplot as plt, plotly.graph_objects as go
from scipy.integrate import quad as iq

A=np.pi/(180*3600); G,c,Ms,Mpc=6.674e-11,299792458,1.989e30,3.086e22

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
    # Physical: b_E = D_L * theta_E (impact parameter at lens plane)
    b_E=DL*tE*A  # in meters
    b_E_kpc=b_E/(1e3*pc)  # in kpc for display
    # Normalized coords: Source(z=0) -> Lens(z=zL_n) -> Observer(z=1)
    zS_n,zL_n,zO_n=0,0.35,1
    # Scale factor for visualization (normalized units)
    scale=0.12  # visual scale
    fig=go.Figure()
    # Source
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[zS_n],mode='markers',marker=dict(size=15,color='yellow'),name='Quelle'))
    # Lens
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[zL_n],mode='markers',marker=dict(size=14,color='red',symbol='square'),name='Linse'))
    # Observer
    fig.add_trace(go.Scatter3d(x=[0],y=[0],z=[zO_n],mode='markers',marker=dict(size=12,color='#5dade2'),name='Beobachter'))
    # Impact parameter circle at LENS PLANE (z = zL_n) - THIS IS b_E, NOT theta_E!
    th=np.linspace(0,2*np.pi,60)
    fig.add_trace(go.Scatter3d(x=scale*np.cos(th),y=scale*np.sin(th),z=[zL_n]*60,mode='lines',line=dict(color='lime',width=4),name=f'Impact circle b_E'))
    # Ray paths with crossing points at lens plane
    angs=np.arctan2(pos[:,1],pos[:,0]) if n>0 else np.array([0,np.pi/2,np.pi,3*np.pi/2])
    rads=np.hypot(pos[:,0],pos[:,1])/(A*tE) if n>0 and tE>0 else np.ones(4)
    b_vals=[]
    for i in range(min(n,4)):
        phi=angs[i]; r_n=min(rads[i],1.2)*scale  # normalized impact param
        bx,by=r_n*np.cos(phi),r_n*np.sin(phi)  # crossing point at lens plane
        b_vals.append(np.hypot(bx,by)/scale)  # ratio to b_E
        # Ray: Source -> lens crossing -> Observer
        fig.add_trace(go.Scatter3d(x=[0,bx,0],y=[0,by,0],z=[zS_n,zL_n,zO_n],mode='lines+markers',line=dict(color=cols[i],width=4),marker=dict(size=[4,8,4],color=cols[i]),name=f'Ray {i+1}'))
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

def t8(zL,zS,tE):
    DL,DS,DLS=cosmo(zL,zS); M=mass(tE,DL,DS,DLS)
    r_s=2*G*M*Ms/c**2  # Schwarzschild radius
    R_E=tE*A*DL  # Einstein radius in meters
    phi=(1+np.sqrt(5))/2  # Golden ratio
    # Xi values at different radii
    r_vals=np.logspace(np.log10(r_s),np.log10(R_E*10),100)
    Xi_weak=r_s/(2*r_vals)  # Weak field Xi
    Xi_strong=1-np.exp(-phi*r_vals/r_s)  # Strong field Xi
    s_weak=1+Xi_weak; s_strong=1+Xi_strong
    D_weak=1/s_weak; D_strong=1/s_strong
    # Special radii
    Xi_RE=r_s/(2*R_E); s_RE=1+Xi_RE; D_RE=1/s_RE
    Xi_rs=r_s/(2*r_s); s_rs=1+Xi_rs; D_rs=1/s_rs
    # PPN lensing angle
    alpha_xi=r_s/R_E  # Xi-only
    alpha_ppn=2*r_s/R_E  # PPN (1+gamma) with gamma=1
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    # Top-left: Xi vs r
    ax=axes[0,0]; ax.loglog(r_vals/r_s,Xi_weak,'b-',lw=2,label='Ξ weak')
    ax.loglog(r_vals/r_s,Xi_strong,'r--',lw=2,label='Ξ strong')
    ax.axvline(R_E/r_s,color='green',ls=':',label=f'R_E={R_E/r_s:.1e} r_s')
    ax.set_xlabel('r/r_s'); ax.set_ylabel('Ξ(r)'); ax.legend(); ax.grid(alpha=.3)
    ax.set_title('Radial Scaling Gauge: Ξ(r)')
    # Top-right: s(r) and D(r)
    ax=axes[0,1]; ax.semilogx(r_vals/r_s,s_weak,'b-',lw=2,label='s(r) weak')
    ax.semilogx(r_vals/r_s,D_weak,'b--',lw=2,label='D(r) weak')
    ax.axhline(1,color='gray',ls=':'); ax.set_xlabel('r/r_s'); ax.set_ylabel('s, D')
    ax.legend(); ax.grid(alpha=.3); ax.set_title('Scaling Factors s(r)=1+Ξ, D(r)=1/s')
    # Bottom-left: Lensing deflection
    ax=axes[1,0]; b_vals=np.linspace(R_E*0.5,R_E*3,50)
    alpha_b=2*r_s/b_vals  # PPN deflection angle
    ax.plot(b_vals/R_E,np.degrees(alpha_b)*3600,'g-',lw=2)
    ax.axvline(1,color='red',ls='--',label='b=R_E')
    ax.set_xlabel('b/R_E'); ax.set_ylabel('α [arcsec]'); ax.grid(alpha=.3)
    ax.set_title('PPN Ablenkungswinkel α=(1+γ)r_s/b'); ax.legend()
    # Bottom-right: Key values table as bar chart
    ax=axes[1,1]; vals=[Xi_RE,s_RE,D_RE,np.degrees(alpha_ppn)*3600]
    names=['Ξ(R_E)','s(R_E)','D(R_E)','α_PPN [as]']
    colors=['blue','green','orange','red']
    bars=ax.bar(names,vals,color=colors); ax.set_ylabel('Value')
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{v:.2e}',ha='center',va='bottom',fontsize=8)
    ax.set_title('RSG Werte am Einstein-Radius'); ax.set_yscale('log')
    plt.tight_layout()
    out=f"## Radial Scaling Gauge\\n\\n| Parameter | Wert |\\n|--|--|\\n| r_s | {r_s:.3e} m |\\n| R_E | {R_E:.3e} m |\\n| R_E/r_s | {R_E/r_s:.2e} |\\n| Ξ(R_E) | {Xi_RE:.2e} |\\n| s(R_E) | {s_RE:.6f} |\\n| D(R_E) | {D_RE:.6f} |\\n| α_Ξ | {np.degrees(alpha_xi)*3600:.4e} as |\\n| α_PPN | {np.degrees(alpha_ppn)*3600:.4e} as |\\n| θ_E | {tE:.4f} as |\\n\\n**Formeln:**\\n- Ξ_weak = r_s/(2r)\\n- s(r) = 1 + Ξ(r)\\n- D(r) = 1/s(r)\\n- α_PPN = (1+γ)r_s/b"
    return out, fig

with gr.Blocks(title='RSG Lensing') as demo:
    gr.Markdown('# RSG Lensing Suite')
    with gr.Tab('Data'):
        txt=gr.Textbox(value=QUAD,lines=5,label='Positions')
        unit=gr.Dropdown(['arcsec','mas'],value='arcsec')
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
        b8.click(t8,[zL,zS,tE],[o8,p8])
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
