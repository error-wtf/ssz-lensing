import json

code = """#@title RSG Lensing UI (6 Tabs) - Run this cell
!pip install -q gradio numpy matplotlib scipy

import gradio as gr, numpy as np, matplotlib.pyplot as plt
from scipy.integrate import quad as iq

A=np.pi/(180*3600); G,c,Ms,Mpc=6.674e-11,3e8,2e30,3.086e22

def parse(t,u='arcsec'):
    f={'arcsec':A,'mas':A/1000,'rad':1}[u]
    return np.array([[float(x) for x in l.replace(',',' ').split()[:2]] for l in t.strip().split('\\n') if l.strip() and not l.startswith('#')])*f

def morph(pos):
    r=np.hypot(pos[:,0],pos[:,1]); rm=np.mean(r); n=len(pos)
    if n==4: return 'QUAD',0.95
    elif n==2: return 'DOUBLE',0.9
    elif n>=8 and np.std(r)/rm<0.15: return 'RING',0.85
    return 'UNKNOWN',0.5

def ring_fit(pos):
    c=np.mean(pos,axis=0); rel=pos-c; r=np.hypot(rel[:,0],rel[:,1])
    R=np.mean(r); rms=np.sqrt(np.mean((r-R)**2))
    return c[0],c[1],R,rms

def invert(pos):
    if len(pos)!=4: return None
    r=np.hypot(pos[:,0],pos[:,1]); tE=np.mean(r)
    return tE, np.sqrt(np.mean((r-tE)**2))

def cosmo(zL,zS):
    def E(z): return 1/np.sqrt(0.315*(1+z)**3+0.685)
    DH=c/(67.4*1000/Mpc)
    cL,_=iq(E,0,zL); cS,_=iq(E,0,zS)
    return cL*DH/(1+zL), cS*DH/(1+zS), (cS-cL)*DH/(1+zS)

def mass(tE,DL,DS,DLS):
    Scr=c**2*DS/(4*np.pi*G*DL*DLS); RE=tE*A*DL
    return np.pi*RE**2*Scr/Ms

QUAD='''0.740, 0.565
-0.635, 0.470
-0.480, -0.755
0.870, -0.195'''

def t1(txt,u):
    pos=parse(txt,u); n=len(pos); r=np.hypot(pos[:,0],pos[:,1])
    fig,ax=plt.subplots(figsize=(5,5)); p=pos/A
    ax.scatter(p[:,0],p[:,1],s=80,c='blue'); ax.set_aspect('equal'); ax.grid(alpha=.3)
    return f"Points: {n}, Mean R: {np.mean(r)/A:.4f} arcsec", fig

def t2(txt,u):
    pos=parse(txt,u); m,conf=morph(pos); cx,cy,R,rms=ring_fit(pos)
    fig,ax=plt.subplots(figsize=(5,5)); p=pos/A; t=np.linspace(0,2*np.pi,100)
    ax.scatter(p[:,0],p[:,1],s=80,c='blue')
    ax.plot(cx/A+R/A*np.cos(t),cy/A+R/A*np.sin(t),'g--',lw=2)
    ax.scatter([cx/A],[cy/A],s=150,c='red',marker='+',lw=3); ax.set_aspect('equal')
    return f"**{m}** ({conf:.0%})\\nR={R/A:.4f} arcsec", fig

def t3(txt,u):
    pos=parse(txt,u); cx,cy,R,rms=ring_fit(pos)
    fig,ax=plt.subplots(figsize=(5,5)); p=pos/A; t=np.linspace(0,2*np.pi,100)
    ax.scatter(p[:,0],p[:,1],s=80,c='blue')
    ax.plot(cx/A+R/A*np.cos(t),cy/A+R/A*np.sin(t),'g-',lw=2)
    ax.scatter([cx/A],[cy/A],s=150,c='red',marker='+',lw=3); ax.set_aspect('equal')
    return f"R={R/A:.4f} arcsec\\nRMS={rms/A:.6f} arcsec", fig

def t4(txt,u):
    pos=parse(txt,u); res=invert(pos)
    if not res: return 'Need 4 pts', None
    tE,rms=res
    fig,ax=plt.subplots(figsize=(5,5)); p=pos/A; t=np.linspace(0,2*np.pi,100)
    ax.scatter(p[:,0],p[:,1],s=100,c='blue',zorder=5)
    ax.plot(tE/A*np.cos(t),tE/A*np.sin(t),'g--',lw=2)
    ax.scatter([0],[0],s=150,c='red',marker='+',lw=3); ax.set_aspect('equal')
    return f"theta_E={tE/A:.4f} arcsec\\nRMS={rms/A:.6f} arcsec", fig

def t5(zL,zS,tE):
    DL,DS,DLS=cosmo(zL,zS); M=mass(tE,DL,DS,DLS)
    fig,ax=plt.subplots(); ax.barh(['D_L','D_S','D_LS'],[DL/Mpc,DS/Mpc,DLS/Mpc])
    ax.set_xlabel('Mpc')
    return f"D_L={DL/Mpc:.0f} Mpc\\nD_S={DS/Mpc:.0f} Mpc\\n**M={M:.2e} Msun**", fig

def t6(zL,zS,tE):
    DL,DS,DLS=cosmo(zL,zS); RE=tE*A*DL
    fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(111,projection='3d')
    ax.scatter([0],[0],[0],s=200,c='yellow',label='Obs')
    ax.scatter([0],[0],[DL/Mpc],s=150,c='red',marker='s',label='Lens')
    t=np.linspace(0,2*np.pi,50)
    ax.plot(RE/Mpc*np.cos(t),RE/Mpc*np.sin(t),[DL/Mpc]*50,'g-',lw=2)
    ax.scatter([0],[0],[DS/Mpc],s=150,c='blue',marker='*',label='Source')
    ax.legend(); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('D (Mpc)')
    return f"R_E={RE/1e3:.1f} kpc", fig

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
"""

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
