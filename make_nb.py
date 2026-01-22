import json

code = '''#@title RSG Lensing Suite - Clone & Run (Full Version)
# Clone the repository and install dependencies
!git clone --depth 1 https://github.com/error-wtf/ssz-lensing.git 2>/dev/null || echo "Repo already cloned"
!pip install -q gradio numpy matplotlib scipy

import sys
sys.path.insert(0, 'ssz-lensing/src')

import gradio as gr, numpy as np, matplotlib.pyplot as plt
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

def t6(zL,zS,tE):
    DL,DS,DLS=cosmo(zL,zS); M=mass(tE,DL,DS,DLS)
    alpha=tE*A  # deflection angle in radians
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    # LEFT: Classic lensing diagram (Source -> Lens -> Observer)
    ax1=axes[0]; ax1.set_facecolor('#1a1a2e')
    # Positions along optical axis (x): Source=0, Lens=DL, Observer=DS
    xS,xL,xO=0,DL/Mpc,DS/Mpc; scale=xO/10
    # True source position (on axis)
    ax1.scatter([xS],[0],s=200,c='yellow',edgecolors='orange',zorder=10,label='Wahre Quelle')
    # Lens (massive object)
    circle=plt.Circle((xL,0),scale*0.8,color='orange',alpha=0.8,zorder=5)
    ax1.add_patch(circle)
    ax1.annotate('Gravitationsfeld',xy=(xL,0),fontsize=9,ha='center',va='center',color='white',weight='bold')
    # Gravitational field ellipse
    from matplotlib.patches import Ellipse
    field=Ellipse((xL,0),scale*4,scale*3,alpha=0.2,color='cyan',zorder=2)
    ax1.add_patch(field)
    # Observer (focal point)
    ax1.scatter([xO],[0],s=150,c='#5dade2',edgecolors='white',zorder=10,label='Brennpunkt (Observer)')
    # Geodesics bending around lens
    deflect=scale*1.5  # visual deflection for clarity
    # Upper ray: Source -> bent around lens -> Observer
    ax1.annotate('',xy=(xL,deflect*0.8),xytext=(xS,deflect*1.5),
                 arrowprops=dict(arrowstyle='-',color='white',lw=1.5,ls='--'))
    ax1.annotate('',xy=(xO,0),xytext=(xL,deflect*0.8),
                 arrowprops=dict(arrowstyle='->',color='white',lw=1.5))
    # Lower ray
    ax1.annotate('',xy=(xL,-deflect*0.8),xytext=(xS,-deflect*1.5),
                 arrowprops=dict(arrowstyle='-',color='white',lw=1.5,ls='--'))
    ax1.annotate('',xy=(xO,0),xytext=(xL,-deflect*0.8),
                 arrowprops=dict(arrowstyle='->',color='white',lw=1.5))
    # Apparent positions (Scheinbarer Ort)
    ax1.scatter([xS],[deflect*1.5],s=120,c='yellow',alpha=0.5,zorder=8)
    ax1.scatter([xS],[-deflect*1.5],s=120,c='yellow',alpha=0.5,zorder=8)
    ax1.annotate('Scheinbarer Ort',xy=(xS-scale*0.5,deflect*1.8),fontsize=8,color='yellow',ha='center')
    ax1.annotate('Scheinbarer Ort',xy=(xS-scale*0.5,-deflect*1.8),fontsize=8,color='yellow',ha='center')
    ax1.annotate('elektromagnetische Wellen',xy=(xL/2,deflect*1.2),fontsize=7,color='lightgray',ha='center')
    ax1.set_xlim(-scale*2,xO+scale); ax1.set_ylim(-scale*4,scale*4)
    ax1.set_aspect('equal'); ax1.axis('off')
    ax1.set_title('Gravitationslinsen-Effekt',fontsize=12,color='white',pad=10)
    # RIGHT: Observer Sky (angular projection)
    ax2=axes[1]
    th=np.linspace(0,2*np.pi,100)
    ax2.plot(tE*np.cos(th),tE*np.sin(th),'g-',lw=3,label=f'Einstein Ring (θ_E={tE:.3f}")')
    ax2.scatter([0],[0],s=100,c='red',marker='+',lw=2,label='Linse (Zentrum)')
    ax2.scatter([0],[0],s=50,c='yellow',marker='*',alpha=0.7,label='Quelle (dahinter)')
    ax2.set_xlim(-tE*2,tE*2); ax2.set_ylim(-tE*2,tE*2)
    ax2.set_aspect('equal'); ax2.grid(alpha=0.3)
    ax2.set_xlabel('θ_x [arcsec]'); ax2.set_ylabel('θ_y [arcsec]')
    ax2.set_title('Beobachter-Himmel: Ring als Winkelprojektion')
    ax2.legend(loc='upper right',fontsize=8)
    plt.tight_layout()
    out=f"## Gravitationslinsen-Geometrie\\n**Kernaussage:** Der Einsteinring ist KEIN physisches Objekt im Raum.\\nEr existiert nur als **Winkelprojektion am Himmel des Beobachters**.\\n\\n| Parameter | Wert |\\n|--|--|\\n| D_L | {DL/Mpc:.1f} Mpc |\\n| D_S | {DS/Mpc:.1f} Mpc |\\n| D_LS | {DLS/Mpc:.1f} Mpc |\\n| θ_E | {tE:.4f} arcsec |\\n| α (Ablenkung) | {np.degrees(alpha)*3600:.4f} arcsec |\\n| M | {M:.2e} M☉ |\\n\\n**Links:** Geodäten biegen sich um die Masse\\n**Rechts:** Ring erscheint in Winkelkoordinaten"
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
        b6.click(t6,[zL,zS,tE],[o6,p6])
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

print('Created SSZ_Lensing_Colab.ipynb with 6 tabs')
