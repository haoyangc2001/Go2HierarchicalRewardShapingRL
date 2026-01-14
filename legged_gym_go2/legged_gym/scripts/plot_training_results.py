#!/usr/bin/env python3
"""
Training Log Visualization Script for MCRA_RL Project

This script reads training log files from Reach-Avoid PPO training and generates
visualization plots of training metrics over iterations.

Usage:
    python plot_training_results.py <log_file_path> [options]

Example:
    python plot_training_results.py logs/high_level_go2/20260106-104432/training.log
"""

import os
import sys
import argparse

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_log_file(log_file_path):
    """
    Parse training log file and extract metrics.

    Args:
        log_file_path (str): Path to the training log file

    Returns:
        dict: Dictionary containing lists of extracted metrics
    """
    print(f"Parsing log file: {log_file_path}")

    # Initialize data containers
    data = {
        'iterations': [],
        'success': [],
        'cost': [],
        'policy_loss': [],
        'value_loss': [],
        'Vmean': [],
        'Rmean': [],
        'Vrmse': [],
        'VexpVar': [],
        'adv_std': [],
        'elapsed': []
    }

    expected_keys = [
        'success',
        'cost',
        'policy_loss',
        'value_loss',
        'Vmean',
        'Rmean',
        'Vrmse',
        'VexpVar',
        'adv_std',
        'elapsed',
    ]

    line_count = 0
    parsed_count = 0

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                line_count += 1

                if not line.startswith("iter "):
                    continue

                parts = [p.strip() for p in line.split("|")]
                if not parts:
                    continue

                iter_tokens = parts[0].split()
                if len(iter_tokens) < 2:
                    continue

                try:
                    iteration = int(iter_tokens[1])
                except ValueError:
                    continue

                metrics = {}
                for part in parts[1:]:
                    if not part:
                        continue
                    key_value = part.split(" ", 1)
                    if len(key_value) != 2:
                        continue
                    key, value = key_value
                    value = value.strip()
                    if key == "elapsed":
                        value = value.rstrip("s")
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        continue

                parsed_count += 1
                data['iterations'].append(iteration)
                for key in expected_keys:
                    data[key].append(metrics.get(key, np.nan))

        print(f"Parsed {parsed_count} out of {line_count} lines successfully")

        # Convert to numpy arrays for easier manipulation
        for key in data:
            data[key] = np.array(data[key])

        return data

    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing log file: {e}")
        sys.exit(1)


def plot_success_rate(data, output_dir, log_file_name):
    """
    Plot success rate over iterations.

    Args:
        data (dict): Parsed log data
        output_dir (str): Directory to save the plot
        log_file_name (str): Name of the log file (for plot title)
    """
    iterations = data['iterations']
    success = data['success']

    # Check if we have valid success data
    if len(success) == 0:
        print("Warning: No success rate data to plot")
        return

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot success rate
    plt.plot(iterations, success, 'b-', linewidth=2, label='Success Rate')

    # Add smoothing if enough data points
    if len(success) > 10:
        window_size = min(20, len(success) // 10)
        if window_size >= 3:
            smoothed = np.convolve(success, np.ones(window_size)/window_size, mode='valid')
            smoothed_iterations = iterations[window_size-1:]
            plt.plot(smoothed_iterations, smoothed, 'r--', linewidth=1.5,
                    label=f'Moving Average (window={window_size})')

    # Customize plot
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title(f'Training Success Rate Progress\n{log_file_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Set y-axis limits
    plt.ylim(-0.05, 1.05)

    # Use integer ticks for x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add text annotation for final success rate
    if len(success) > 0:
        final_success = success[-1]
        plt.annotate(f'Final: {final_success:.3f}',
                    xy=(1, final_success),
                    xytext=(0.95, 0.95),
                    textcoords='axes fraction',
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Save figure
    output_path = os.path.join(output_dir, f'{log_file_name}_success_rate.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Success rate plot saved to: {output_path}")

    # Also save as PDF for vector graphics
    pdf_path = os.path.join(output_dir, f'{log_file_name}_success_rate.pdf')
    plt.savefig(pdf_path, format='pdf')
    print(f"Success rate plot (PDF) saved to: {pdf_path}")

    plt.close()


def plot_multiple_metrics(data, output_dir, log_file_name):
    """
    Plot multiple training metrics in subplots.

    Args:
        data (dict): Parsed log data
        output_dir (str): Directory to save the plot
        log_file_name (str): Name of the log file (for plot title)
    """
    iterations = data['iterations']

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Training Metrics Overview\n{log_file_name}', fontsize=16, fontweight='bold')

    # Plot 1: Success Rate
    ax = axes[0, 0]
    if len(data['success']) > 0:
        ax.plot(iterations, data['success'], 'b-', linewidth=1.5)
        ax.set_ylabel('Success Rate', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title('Success Rate Progress', fontsize=11)

    # Plot 2: Execution Cost
    ax = axes[0, 1]
    if len(data['cost']) > 0 and not np.all(np.isnan(data['cost'])):
        ax.plot(iterations, data['cost'], 'g-', linewidth=1.5)
        ax.set_ylabel('Execution Cost', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Execution Cost (Avg. Timesteps to Target)', fontsize=11)

    # Plot 3: Policy and Value Loss
    ax = axes[1, 0]
    if len(data['policy_loss']) > 0 and not np.all(np.isnan(data['policy_loss'])):
        ax.plot(iterations, data['policy_loss'], 'r-', linewidth=1.5, label='Policy Loss')
    if len(data['value_loss']) > 0 and not np.all(np.isnan(data['value_loss'])):
        ax.plot(iterations, data['value_loss'], 'm-', linewidth=1.5, label='Value Loss')
        ax.set_ylabel('Loss', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Policy and Value Loss', fontsize=11)
        ax.legend(fontsize=8)
        # Use log scale if value loss is large
        if np.nanmax(data['value_loss']) > 1000:
            ax.set_yscale('log')

    # Plot 4: Value Statistics
    ax = axes[1, 1]
    if len(data['Vmean']) > 0 and not np.all(np.isnan(data['Vmean'])):
        ax.plot(iterations, data['Vmean'], 'c-', linewidth=1.5, label='V mean')
    if len(data['Rmean']) > 0 and not np.all(np.isnan(data['Rmean'])):
        ax.plot(iterations, data['Rmean'], 'y-', linewidth=1.5, label='R mean')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Value Function Statistics', fontsize=11)
        ax.legend(fontsize=8)

    # Plot 5: Explained Variance and RMSE
    ax = axes[2, 0]
    if len(data['VexpVar']) > 0 and not np.all(np.isnan(data['VexpVar'])):
        ax.plot(iterations, data['VexpVar'], 'b-', linewidth=1.5, label='Explained Variance')
        ax.set_ylabel('Explained Variance', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Value Function Explained Variance', fontsize=11)
        ax.set_ylim(-0.1, 1.1)

    # Plot 6: Advantage Standard Deviation
    ax = axes[2, 1]
    if len(data['adv_std']) > 0 and not np.all(np.isnan(data['adv_std'])):
        ax.plot(iterations, data['adv_std'], 'orange', linewidth=1.5)
        ax.set_ylabel('Advantage Std Dev', fontsize=10)
        ax.set_xlabel('Iteration', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title('Advantage Standard Deviation', fontsize=11)

    # Set x-labels for bottom row
    for ax in axes[2, :]:
        ax.set_xlabel('Iteration', fontsize=10)

    # Use integer ticks for x-axis
    for ax_row in axes:
        for ax in ax_row:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, f'{log_file_name}_all_metrics.png')
    plt.savefig(output_path, dpi=300)
    print(f"All metrics plot saved to: {output_path}")

    # Also save as PDF
    pdf_path = os.path.join(output_dir, f'{log_file_name}_all_metrics.pdf')
    plt.savefig(pdf_path, format='pdf')
    print(f"All metrics plot (PDF) saved to: {pdf_path}")

    plt.close()


def generate_summary_report(data, output_dir, log_file_name):
    """
    Generate a text summary report of training statistics.

    Args:
        data (dict): Parsed log data
        output_dir (str): Directory to save the report
        log_file_name (str): Name of the log file
    """
    report_path = os.path.join(output_dir, f'{log_file_name}_summary.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Training Log Analysis Report\n")
        f.write(f"============================\n\n")
        f.write(f"Log File: {log_file_name}\n")
        f.write(f"Total Iterations: {len(data['iterations'])}\n\n")

        # Success rate statistics
        if len(data['success']) > 0:
            f.write("Success Rate Statistics:\n")
            f.write(f"  Initial: {data['success'][0]:.3f}\n")
            f.write(f"  Final: {data['success'][-1]:.3f}\n")
            f.write(f"  Maximum: {np.nanmax(data['success']):.3f}\n")
            f.write(f"  Minimum: {np.nanmin(data['success']):.3f}\n")
            f.write(f"  Average: {np.nanmean(data['success']):.3f}\n")
            f.write(f"  Std Dev: {np.nanstd(data['success']):.3f}\n\n")

        # Execution cost statistics
        if len(data['cost']) > 0 and not np.all(np.isnan(data['cost'])):
            f.write("Execution Cost Statistics:\n")
            f.write(f"  Final: {data['cost'][-1]:.1f}\n")
            f.write(f"  Average: {np.nanmean(data['cost']):.1f}\n")
            f.write(f"  Minimum: {np.nanmin(data['cost']):.1f}\n")
            f.write(f"  Maximum: {np.nanmax(data['cost']):.1f}\n\n")

        # Training time statistics
        if len(data['elapsed']) > 0 and not np.all(np.isnan(data['elapsed'])):
            total_time = np.nansum(data['elapsed'])
            avg_time = np.nanmean(data['elapsed'])
            f.write("Training Time Statistics:\n")
            f.write(f"  Total: {total_time:.1f} seconds ({total_time/3600:.2f} hours)\n")
            f.write(f"  Average per iteration: {avg_time:.1f} seconds\n")
            f.write(f"  Estimated iterations per hour: {3600/avg_time:.1f}\n\n")

        # Final values of other metrics
        f.write("Final Iteration Values:\n")
        metrics = ['policy_loss', 'value_loss', 'Vmean', 'Rmean', 'Vrmse', 'VexpVar', 'adv_std']
        for metric in metrics:
            if len(data[metric]) > 0 and not np.all(np.isnan(data[metric])):
                f.write(f"  {metric}: {data[metric][-1]:.6f}\n")

    print(f"Summary report saved to: {report_path}")


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description='Generate visualization plots from MCRA_RL training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s logs/high_level_go2/20260106-104432/training.log
  %(prog)s /path/to/training.log --all-metrics
  %(prog)s /path/to/training.log --output-dir /custom/output/path
        """
    )

    parser.add_argument('log_file', type=str, default = 'logs/high_level_go2/20260106-104432/training.log', help='Path to the training log file')
    parser.add_argument('--output-dir', type=str, help='Directory to save plots (default: same as log file)')
    parser.add_argument('--all-metrics', action='store_true', help='Generate comprehensive plots of all metrics')
    parser.add_argument('--no-summary', action='store_true', help='Skip generating summary report')

    args = parser.parse_args()

    # Validate log file path
    log_file_path = Path(args.log_file)
    if not log_file_path.exists():
        print(f"Error: Log file not found: {args.log_file}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = log_file_path.parent

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get log file name without extension for naming output files
    log_file_name = log_file_path.stem

    # Parse log file
    data = parse_log_file(str(log_file_path))

    if len(data['iterations']) == 0:
        print("Error: No data parsed from log file")
        sys.exit(1)

    print(f"\nGenerating plots for {len(data['iterations'])} iterations...")

    # Generate success rate plot (always generated)
    plot_success_rate(data, str(output_dir), log_file_name)

    # Generate comprehensive plots if requested
    if args.all_metrics:
        plot_multiple_metrics(data, str(output_dir), log_file_name)

    # Generate summary report unless disabled
    if not args.no_summary:
        generate_summary_report(data, str(output_dir), log_file_name)

    print(f"\nAll plots and reports saved to: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()
