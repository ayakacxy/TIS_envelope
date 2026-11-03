"""Draw CPU, GPU, and paired comparison figures from server measurements."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.plot_cpu_gpu import (FIGURE_WIDTH, _annotate_endpoints, _annotate_fastest_bars,
                                     _axis_style, _contrast_color, _row_legend_order, _save, _setup_axis,
                                     _set_typeface)

STYLES = {
    'numpy': ('NumPy', '#AAB5C3', 'o', '-'),
    'numba_serial': ('Numba (1T)', '#85B9D6', 's', '-'),
    'numba_10t': ('Numba (10T)', '#5A9DCB', '^', '--'),
    'numba_32t': ('Numba (32T)', '#3977A8', 'P', '-.'),
    'jax_cpu': ('JAX', '#A3B86C', 'D', '-.'),
    'cupy_existing': ('CuPy (unscaled)', '#F0C47F', 'v', '-'),
    'cupy_stable': ('CuPy (scaled)', '#72BFB0', 'o', '-'),
    'jax_cuda': ('JAX', '#A3B86C', 'D', '-.'),
    'torch_cuda': ('PyTorch eager', '#DCB2C6', 'D', '-.'),
    'torch_cuda_compiled': ('PyTorch compiled', '#AE6E94', 'X', '--'),
}
CPU = ('numpy', 'numba_serial', 'numba_32t', 'jax_cpu')
GPU = ('cupy_existing', 'cupy_stable', 'jax_cuda', 'torch_cuda', 'torch_cuda_compiled')


def read_reports(paths):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if [r['metadata']['benchmark_device'] for r in reports] != ['cpu','cpu','gpu']:
        raise ValueError('expected CPU baselines, CPU 32T, GPU files in that order')
    if [reports[i]['metadata']['numba_parallel_threads'] for i in (0,1)] != [None,32]:
        raise ValueError('expected baseline and 32T CPU measurements')
    if reports[0]['metadata']['cpu_affinity'] != reports[1]['metadata']['cpu_affinity']:
        raise ValueError('CPU baseline and 32T affinity differs')
    meta = reports[0]['metadata']
    for report in reports[1:]:
        for key in ('cpu_model','gpu','machine','numpy','numba','jax','jaxlib','torch','cupy',
                    'dtype','warmups','repeats','seed','synthetic','reference','source_sha256',
                    'existing_cupy_source_sha256'):
            if report['metadata'][key] != meta[key]:
                raise ValueError(f'inconsistent metadata: {key}')
    index = {}
    for report in reports:
        for record in report['results']:
            key = (record['dataset'],record['n'],record['backend'])
            if key in index:
                raise ValueError(f'duplicate record {key}')
            index[key] = record
    cases = {(d,n) for d,n,b in index if b == 'numpy'}
    for d,n in cases:
        hashes = set()
        for backend in CPU + GPU:
            item = index.get((d,n,backend))
            if item is None:
                raise ValueError(f'missing {backend} for {d} {n}')
            hashes.add(item['input_sha256'])
            if item['median_ms'] <= 0 or len(item['samples_ms']) != meta['repeats']:
                raise ValueError(f'invalid samples for {backend} at {d} {n}')
        if len(hashes) != 1:
            raise ValueError(f'mismatched inputs for {d} {n}')
    return index, meta


def sizes_for(index):
    return sorted(n for d,n,b in index if d == 'synthetic' and b == 'numpy')


def rate(index, d, n, b):
    return 1000 / index[(d,n,b)]['median_ms']


def grouped_legend(fig, groups, x, y, columns=2):
    order = _row_legend_order(len(groups), columns)
    handles = [Patch(facecolor=STYLES[groups[i]][1], label=STYLES[groups[i]][0]) for i in order]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(x,y), ncol=columns,
               frameon=False, fontsize=7.0, handlelength=0.9, columnspacing=1.0,
               handletextpad=0.35, borderaxespad=0)


def comparison(index, output):
    sizes = sizes_for(index)
    real = sorted((d,n) for d,n,b in index if d != 'synthetic' and b == 'numpy')
    if len(real) != 1:
        raise ValueError('expected exactly one real-head dataset')
    dataset, n = real[0]
    methods = CPU + GPU
    fig,(a,b)=plt.subplots(2,1,figsize=(FIGURE_WIDTH,6.2), gridspec_kw={'height_ratios':[1.03,1]})
    fig.patch.set_facecolor('white')
    fig.text(0.105,0.975,'Server CPU vs GPU',fontsize=11,weight='bold',va='top')
    fig.text(0.97,0.975,'Xeon Gold 6230  |  RTX 2080 Ti',fontsize=8.0,color='#505050',va='top',ha='right')
    fig.text(0.10,0.925,'CPU',fontsize=7.6,weight='bold',va='top')
    fig.text(0.54,0.925,'GPU',fontsize=7.6,weight='bold',va='top')
    grouped_legend(fig,CPU,0.10,0.902)
    grouped_legend(fig,GPU,0.54,0.902)
    fig.subplots_adjust(left=0.12,right=0.97,bottom=0.13,top=0.79,hspace=0.28)
    pos=np.arange(len(sizes))
    _setup_axis(a,'(a) Synthetic data','Calls/s (symlog)',sizes)
    width=.88/len(methods)
    bars=[]
    max_rate=0
    for i,name in enumerate(methods):
        values=[rate(index,'synthetic',size,name) for size in sizes]
        max_rate=max(max_rate,max(values))
        bars.append(a.bar(pos-.44+width*(i+.5),values,width=width,color=STYLES[name][1],edgecolor='white',linewidth=.25))
    a.set_yscale('symlog',linthresh=1)
    a.set_ylim(0,max_rate*4)
    _annotate_fastest_bars(a,bars)
    b.set_title('(b) Real head',loc='left',pad=5,fontsize=9.0,fontweight='bold')
    b.set_title(f'{n/1_000_000:.2f}M voxels',loc='right',pad=5,fontsize=7.8,color='#505050')
    values=[rate(index,dataset,n,name) for name in methods]
    bars=b.bar(np.arange(len(methods)),values,width=.62,color=[STYLES[name][1] for name in methods])
    def label(name):
        if name.startswith('numba_'):
            return 'Numba\n('+STYLES[name][0][7:-1]+')'
        if name.startswith('torch_'):
            return 'PyTorch\n'+('compiled' if name.endswith('compiled') else 'eager')
        if name.startswith('cupy_'):
            return 'CuPy\n('+('unscaled' if name=='cupy_existing' else 'scaled')+')'
        return STYLES[name][0]+'\n('+('CPU' if name in CPU else 'GPU')+')'
    b.set_xticks(np.arange(len(methods)),[label(name) for name in methods])
    b.set_ylabel('Calls/s (symlog)',labelpad=3)
    b.set_yscale('symlog',linthresh=1)
    b.set_ylim(0,max(values)*4)
    _axis_style(b,'y')
    b.tick_params(axis='x',labelsize=6.5,length=0,pad=4)
    b.axvline(len(CPU)-.5,color='#B8B8B8',linewidth=.6,linestyle=':')
    for bar,value in zip(bars,values):
        b.annotate(f'{value:,.0f}',xy=(bar.get_x()+bar.get_width()/2,value),xytext=(0,3),
                   textcoords='offset points',ha='center',va='bottom',color='#202020',fontsize=7.3,fontweight='bold')
    _save(fig,output,'server_cpu_gpu_comparison')


def detail(index, backends, device, output):
    sizes=sizes_for(index)
    pos=np.arange(len(sizes))
    fig=plt.figure(figsize=(FIGURE_WIDTH,6.4))
    grid=fig.add_gridspec(3,2,height_ratios=[1,1,.9])
    a=fig.add_subplot(grid[0,:]);b=fig.add_subplot(grid[1,:]);c=fig.add_subplot(grid[2,0]);d=fig.add_subplot(grid[2,1])
    fig.patch.set_facecolor('white')
    fig.text(.10,.975,'Server '+device,fontsize=11,weight='bold',va='top')
    fig.text(.97,.975,'Intel Xeon Gold 6230  |  FP32' if device=='CPU' else 'NVIDIA RTX 2080 Ti  |  FP32',
             fontsize=8.2,color='#505050',va='top',ha='right')
    handles=[]
    for name in backends:
        label,color,marker,ls=STYLES[name]
        handles.append(Line2D([0],[0],color=_contrast_color(color,3),marker=marker,markerfacecolor=color,
                              linestyle=ls,linewidth=1.3,markersize=4,label=label))
    order=_row_legend_order(len(handles),3)
    fig.legend(handles=[handles[i] for i in order],loc='upper left',bbox_to_anchor=(.10,.94),ncol=3,
               frameon=False,fontsize=7.3,handlelength=1.2,columnspacing=1.5,handletextpad=.35,borderaxespad=0)
    fig.subplots_adjust(left=.12,right=.98,bottom=.085,top=.85,wspace=.31,hspace=.32)
    _setup_axis(a,'(a) Throughput','Calls/s',sizes);a.set_xlabel('')
    width=.8/len(backends);bars=[];highest=0
    for i,name in enumerate(backends):
        values=[rate(index,'synthetic',n,name) for n in sizes];highest=max(highest,max(values))
        bars.append(a.bar(pos-.4+width*(i+.5),values,width=width,color=STYLES[name][1],edgecolor='white',linewidth=.25))
    a.set_yscale('symlog',linthresh=1);a.set_ylim(0,highest*12);_annotate_fastest_bars(a,bars)
    _setup_axis(b,'(b) Time per call','Time (ms)',sizes);b.set_xlabel('');b.set_yscale('log')
    endpoints=[]
    for name in backends:
        _,color,marker,ls=STYLES[name]
        values=[index[('synthetic',n,name)]['median_ms'] for n in sizes]
        b.plot(pos,values,color=_contrast_color(color,3),marker=marker,markerfacecolor=color,
               linestyle=ls,linewidth=1.35,markersize=3.6)
        endpoints.append((values[-1],color))
    b.set_xlim(-.45,len(sizes)+.6);b.margins(y=.18);_annotate_endpoints(b,endpoints,pos[-1])
    _setup_axis(c,'(c) Error vs NumPy FP64','Error (V/m)',sizes);c.set_yscale('log')
    maxerr=[max(index[('synthetic',n,name)]['max_abs_error_vs_float64'] for name in backends) for n in sizes]
    meanerr=[max(index[('synthetic',n,name)]['mean_abs_error_vs_float64'] for name in backends) for n in sizes]
    c.plot(pos,maxerr,'o-',color='#8D72A8',linewidth=1.4,markersize=3.8,label='Max (worst)')
    c.plot(pos,meanerr,'s-',color='#4E978A',linewidth=1.4,markersize=3.8,label='Mean (worst)')
    positive=[v for v in maxerr+meanerr if v>0]
    if positive:c.set_ylim(min(positive)/1.7,max(positive)*4)
    c.legend(loc='upper left',frameon=False,fontsize=7.4,handlelength=1.6,handletextpad=.4,borderaxespad=.35)
    base='numpy' if device=='CPU' else 'torch_cuda'
    targets=('numba_32t','jax_cpu') if device=='CPU' else tuple(x for x in GPU if x != base)
    _setup_axis(d,'(d) Speedup vs NumPy' if device=='CPU' else '(d) Speedup vs PyTorch eager',
                'Speedup (×, log)',sizes)
    d.set_yscale('log');d.axhline(1,color='#333333',linewidth=.75,linestyle='--')
    endpoints=[]; all_values=[]
    for name in targets:
        _,color,marker,ls=STYLES[name]
        values=[index[('synthetic',n,base)]['median_ms']/index[('synthetic',n,name)]['median_ms'] for n in sizes]
        all_values.extend(values)
        d.plot(pos,values,color=_contrast_color(color,3),marker=marker,markerfacecolor=color,
               linestyle=ls,linewidth=1.35,markersize=3.6)
        endpoints.append((values[-1],color))
    d.set_xlim(-.45,len(sizes)+.65);d.set_ylim(min(.75,min(all_values)*.8),max(all_values)*1.35)
    d.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:g}'))
    _annotate_endpoints(d,endpoints,pos[-1])
    _save(fig,output,'server_'+device.lower()+'_performance')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu',required=True);p.add_argument('--cpu32',required=True);p.add_argument('--gpu',required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    index,_=read_reports([args.cpu,args.cpu32,args.gpu])
    plt.rcParams.update({'font.size':8.2,'axes.labelsize':8.2,
                         'axes.titlesize':9.2,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
                         'pdf.fonttype':42,'ps.fonttype':42,'savefig.transparent':False})
    _set_typeface()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    comparison(index,output);detail(index,CPU,'CPU',output);detail(index,GPU,'GPU',output)

if __name__=='__main__':
    main()
