#!/usr/bin/env python
"""
RSG Lensing CLI - User-friendly command-line interface.

Commands:
    import      Import observation data from CSV/JSON
    new-case    Create new case interactively
    run-case    Run full analysis pipeline
    compare     Compare multiple runs
    plot        Generate plots from existing run

Usage:
    python -m ui.cli import data/observations/Q2237+0305
    python -m ui.cli new-case --name MyLens
    python -m ui.cli run-case Q2237+0305 --include-m4
    python -m ui.cli compare runs/run1 runs/run2
    python -m ui.cli plot runs/20250122_Q2237+0305

Authors: Carmen N. Wrede, Lino P. Casu
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline.run_case import CasePipeline
from observables.realdata import load_observation_pack, list_available_systems


def cmd_import(args):
    """Import observation data."""
    print(f"Importing from: {args.path}")
    
    try:
        bundle = load_observation_pack(args.path)
        print(f"Loaded: {bundle.name}")
        print(f"  Images: {bundle.total_images}")
        print(f"  Constraints: {bundle.count_constraints()}")
        if bundle.metadata:
            for k, v in bundle.metadata.items():
                print(f"  {k}: {v}")
        print("\nImport successful!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


def cmd_new_case(args):
    """Create new case interactively."""
    print(f"Creating new case: {args.name}")
    
    data_dir = Path("data/observations") / args.name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nEnter image positions (x y) for each image.")
    print("Type 'done' when finished.\n")
    
    images = []
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
    
    for i, label in enumerate(labels):
        try:
            inp = input(f"Image {label} (x y): ").strip()
            if inp.lower() == 'done':
                break
            x, y = map(float, inp.split())
            images.append((x, y, label))
        except ValueError:
            print("Invalid input. Use: x y (e.g., 0.74 0.56)")
            continue
        except EOFError:
            break
    
    if len(images) < 2:
        print("Need at least 2 images.")
        return 1
    
    # Write CSV
    csv_path = data_dir / "images.csv"
    with open(csv_path, 'w') as f:
        f.write("x,y,sigma_x,sigma_y,image_id,source_ref,epoch\n")
        for x, y, label in images:
            f.write(f"{x},{y},0.01,0.01,{label},user_input,2025.0\n")
    
    # Write meta.json
    meta_path = data_dir / "meta.json"
    with open(meta_path, 'w') as f:
        import json
        json.dump({
            "system_name": args.name,
            "n_images": len(images),
            "notes": "Created via CLI"
        }, f, indent=2)
    
    print(f"\nCreated: {csv_path}")
    print(f"Created: {meta_path}")
    print(f"\nRun analysis with: python -m ui.cli run-case {args.name}")
    return 0


def cmd_run_case(args):
    """Run full analysis pipeline."""
    print(f"Running case: {args.system}")
    print(f"  Include m4: {args.include_m4}")
    print(f"  Withhold: {args.withhold or 'none'}")
    print()
    
    # Find data
    data_path = Path("data/observations") / args.system
    if not data_path.exists():
        print(f"Error: System not found at {data_path}")
        return 1
    
    # Run pipeline
    pipeline = CasePipeline(
        system_name=args.system,
        data_path=str(data_path),
        include_m4=args.include_m4,
        withhold=args.withhold
    )
    
    result = pipeline.run()
    
    print(f"\nRun complete!")
    print(f"Artifacts saved to: {result['run_dir']}")
    print(f"\nSummary:")
    print(f"  Models run: {result['n_models']}")
    print(f"  Best model: {result['best_model']}")
    print(f"  Best residual: {result['best_residual']:.4e}")
    
    return 0


def cmd_compare(args):
    """Compare multiple runs."""
    print(f"Comparing runs:")
    for p in args.runs:
        print(f"  - {p}")
    
    # Load and compare
    from model_zoo.artifacts import load_run_artifacts
    
    results = []
    for run_path in args.runs:
        try:
            data = load_run_artifacts(run_path)
            results.append((run_path, data))
        except Exception as e:
            print(f"Error loading {run_path}: {e}")
    
    if len(results) < 2:
        print("Need at least 2 valid runs to compare.")
        return 1
    
    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    for path, data in results:
        print(f"\n{Path(path).name}:")
        for model, info in data.get('models', {}).items():
            if info.get('success'):
                print(f"  {model}: residual={info.get('max_residual', 'N/A'):.4e}")
    
    return 0


def cmd_plot(args):
    """Generate plots from existing run."""
    print(f"Generating plots from: {args.run_dir}")
    
    run_path = Path(args.run_dir)
    if not run_path.exists():
        print(f"Error: Run not found at {run_path}")
        return 1
    
    # Import visualizer
    from pipeline.stages.visualize import generate_all_plots
    
    output_dir = run_path / "figures"
    generate_all_plots(run_path, output_dir)
    
    print(f"Plots saved to: {output_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='rsg-lensing',
        description='RSG Lensing Inversion - User-friendly CLI'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # import
    p_import = subparsers.add_parser('import', help='Import observation data')
    p_import.add_argument('path', help='Path to observation directory')
    
    # new-case
    p_new = subparsers.add_parser('new-case', help='Create new case')
    p_new.add_argument('--name', required=True, help='System name')
    
    # run-case
    p_run = subparsers.add_parser('run-case', help='Run analysis')
    p_run.add_argument('system', help='System name')
    p_run.add_argument('--include-m4', action='store_true', 
                       help='Include m4 multipole models')
    p_run.add_argument('--withhold', help='Withhold observable (e.g., image:D)')
    
    # compare
    p_cmp = subparsers.add_parser('compare', help='Compare runs')
    p_cmp.add_argument('runs', nargs='+', help='Run directories')
    
    # plot
    p_plot = subparsers.add_parser('plot', help='Generate plots')
    p_plot.add_argument('run_dir', help='Run directory')
    
    args = parser.parse_args()
    
    if args.command == 'import':
        return cmd_import(args)
    elif args.command == 'new-case':
        return cmd_new_case(args)
    elif args.command == 'run-case':
        return cmd_run_case(args)
    elif args.command == 'compare':
        return cmd_compare(args)
    elif args.command == 'plot':
        return cmd_plot(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
