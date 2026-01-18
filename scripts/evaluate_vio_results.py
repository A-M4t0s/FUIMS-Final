#!/usr/bin/env python3
"""
VIO Results Offline Evaluator
Interactive tool for evaluating VIO trajectory results from CSV files.
"""

import os
import sys
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import argparse
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import shutil

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_console():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

class InteractiveTrajectoryViewer:
    """Interactive GUI for trajectory visualization."""
    
    def __init__(self, data: Dict, metrics: Dict):
        self.data = data
        self.metrics = metrics
        self.current_idx = 0
        self.num_poses = metrics['num_poses']
        
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('VIO Trajectory Interactive Viewer', fontsize=16, fontweight='bold')
        
        # Main 3D plot
        self.ax_3d = self.fig.add_subplot(2, 3, (1, 4), projection='3d')
        
        # Side plots
        self.ax_error = self.fig.add_subplot(2, 3, 2)
        self.ax_xyz = self.fig.add_subplot(2, 3, 3)
        self.ax_info = self.fig.add_subplot(2, 3, (5, 6))
        self.ax_info.axis('off')
        
        # Plot trajectories
        self.ax_3d.plot(data['est_x'], data['est_y'], data['est_z'], 
                       'b-', label='Estimated', linewidth=2, alpha=0.6)
        self.ax_3d.plot(data['gt_x'], data['gt_y'], data['gt_z'], 
                       'r--', label='Ground Truth', linewidth=2, alpha=0.6)
        
        # Current pose marker
        self.scatter_est = self.ax_3d.scatter([], [], [], c='blue', s=100, marker='o', label='Current Est')
        self.scatter_gt = self.ax_3d.scatter([], [], [], c='red', s=100, marker='x', label='Current GT')
        
        self.ax_3d.set_xlabel('X (m)', fontweight='bold')
        self.ax_3d.set_ylabel('Y (m)', fontweight='bold')
        self.ax_3d.set_zlabel('Z (m)', fontweight='bold')
        self.ax_3d.set_title('3D Trajectory', fontweight='bold')
        self.ax_3d.legend(loc='upper left')
        self.ax_3d.grid(True, alpha=0.3)
        
        # Error plot setup
        self.line_error, = self.ax_error.plot([], [], 'g-', linewidth=2, label='Error')
        self.vline_error = self.ax_error.axvline(x=0, color='orange', linestyle='--', alpha=0.7)
        self.ax_error.set_xlabel('Pose Index', fontweight='bold')
        self.ax_error.set_ylabel('Error (m)', fontweight='bold')
        self.ax_error.set_title('Position Error', fontweight='bold')
        self.ax_error.grid(True, alpha=0.3)
        self.ax_error.legend()
        
        # XYZ plot setup
        self.line_x, = self.ax_xyz.plot([], [], 'r-', linewidth=1.5, alpha=0.7, label='X')
        self.line_y, = self.ax_xyz.plot([], [], 'g-', linewidth=1.5, alpha=0.7, label='Y')
        self.line_z, = self.ax_xyz.plot([], [], 'b-', linewidth=1.5, alpha=0.7, label='Z')
        self.vline_xyz = self.ax_xyz.axvline(x=0, color='orange', linestyle='--', alpha=0.7)
        self.ax_xyz.set_xlabel('Pose Index', fontweight='bold')
        self.ax_xyz.set_ylabel('Coordinate (m)', fontweight='bold')
        self.ax_xyz.set_title('Estimated Coordinates', fontweight='bold')
        self.ax_xyz.grid(True, alpha=0.3)
        self.ax_xyz.legend()
        
        # Info text
        self.info_text = self.ax_info.text(0.05, 0.95, '', transform=self.ax_info.transAxes,
                                          fontfamily='monospace', fontsize=10, verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Slider for pose selection
        ax_slider = self.fig.add_axes([0.2, 0.05, 0.5, 0.03])
        self.slider = Slider(ax_slider, 'Pose', 0, self.num_poses - 1, valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider_changed)
        
        # Buttons
        ax_prev = self.fig.add_axes([0.15, 0.12, 0.05, 0.04])
        btn_prev = Button(ax_prev, '← Prev')
        btn_prev.on_clicked(self._on_prev)
        
        ax_next = self.fig.add_axes([0.75, 0.12, 0.05, 0.04])
        btn_next = Button(ax_next, 'Next →')
        btn_next.on_clicked(self._on_next)
        
        ax_first = self.fig.add_axes([0.05, 0.12, 0.05, 0.04])
        btn_first = Button(ax_first, '|◀')
        btn_first.on_clicked(self._on_first)
        
        ax_last = self.fig.add_axes([0.85, 0.12, 0.05, 0.04])
        btn_last = Button(ax_last, '▶|')
        btn_last.on_clicked(self._on_last)
        
        self.update_display()
    
    def _on_slider_changed(self, val):
        self.current_idx = int(self.slider.val)
        self.update_display()
    
    def _on_prev(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.slider.set_val(self.current_idx)
    
    def _on_next(self, event):
        if self.current_idx < self.num_poses - 1:
            self.current_idx += 1
            self.slider.set_val(self.current_idx)
    
    def _on_first(self, event):
        self.current_idx = 0
        self.slider.set_val(self.current_idx)
    
    def _on_last(self, event):
        self.current_idx = self.num_poses - 1
        self.slider.set_val(self.current_idx)
    
    def update_display(self):
        """Update all plots and info."""
        idx = self.current_idx
        
        # Update 3D scatter
        est_pos = np.array([self.data['est_x'][idx], self.data['est_y'][idx], self.data['est_z'][idx]])
        gt_pos = np.array([self.data['gt_x'][idx], self.data['gt_y'][idx], self.data['gt_z'][idx]])
        
        self.scatter_est.set_offsets([est_pos[:2]])
        self.scatter_est._offsets3d = ([est_pos[0]], [est_pos[1]], [est_pos[2]])
        self.scatter_gt.set_offsets([gt_pos[:2]])
        self.scatter_gt._offsets3d = ([gt_pos[0]], [gt_pos[1]], [gt_pos[2]])
        
        # Update error plot
        self.line_error.set_data(range(len(self.metrics['errors'])), self.metrics['errors'])
        self.ax_error.set_xlim(-1, self.num_poses)
        self.ax_error.set_ylim(0, self.metrics['ape_max'] * 1.1)
        self.vline_error.set_xdata(idx)
        
        # Update XYZ plot
        self.line_x.set_data(range(len(self.data['est_x'])), self.data['est_x'])
        self.line_y.set_data(range(len(self.data['est_y'])), self.data['est_y'])
        self.line_z.set_data(range(len(self.data['est_z'])), self.data['est_z'])
        self.ax_xyz.set_xlim(-1, self.num_poses)
        y_min = min(self.data['est_x'].min(), self.data['est_y'].min(), self.data['est_z'].min())
        y_max = max(self.data['est_x'].max(), self.data['est_y'].max(), self.data['est_z'].max())
        self.ax_xyz.set_ylim(y_min - 5, y_max + 5)
        self.vline_xyz.set_xdata(idx)
        
        # Compute error
        error = np.linalg.norm(est_pos - gt_pos)
        error_ratio = error / self.metrics['ape_max']
        
        # Update info text
        info_str = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ POSE {idx + 1}/{self.num_poses} ({(idx+1)/self.num_poses*100:.1f}%)
╠══════════════════════════════════════════════════════════════════════════════╣
║ Timestamp: {self.data['timestamp'][idx]:.6f} s
║
║ ESTIMATED POSITION          │ GROUND TRUTH POSITION
║ ────────────────────────    │ ─────────────────────────
║ X: {est_pos[0]:>10.4f} m       │ X: {gt_pos[0]:>10.4f} m
║ Y: {est_pos[1]:>10.4f} m       │ Y: {gt_pos[1]:>10.4f} m
║ Z: {est_pos[2]:>10.4f} m       │ Z: {gt_pos[2]:>10.4f} m
║
║ POSITION ERROR: {error:.4f} m ({error_ratio*100:.1f}% of max)
╠══════════════════════════════════════════════════════════════════════════════╣
║ GLOBAL METRICS
║ ─────────────
║ RMSE: {self.metrics['ape_rmse']:.4f} m  │  MAX: {self.metrics['ape_max']:.4f} m  │  MEAN: {self.metrics['ape_mean']:.4f} m
║ STD:  {self.metrics['ape_std']:.4f} m  │  MIN: {self.metrics['ape_min']:.4f} m  │  Duration: {self.metrics['duration']:.2f}s
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        self.info_text.set_text(info_str)
        
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Display the interactive viewer."""
        plt.tight_layout(rect=[0, 0.15, 1, 0.96])
        plt.show()

class VIOResultsEvaluator:
    """Main evaluation class for VIO results."""
    
    def __init__(self, results_dir: str = "/home/tony/Desktop/MEEC-SA/2º Ano/FUIMS/Resultados"):
        self.results_dir = results_dir
        self.csv_files = []
        self.current_file = None
        self.data = None
        self.parameters = {}
        self.metrics_cache = {}
        
    def list_results_files(self) -> List[str]:
        """List all CSV files in the results directory."""
        if not os.path.exists(self.results_dir):
            print(f"{Colors.FAIL}Results directory not found: {self.results_dir}{Colors.ENDC}")
            return []
        
        csv_files = glob.glob(os.path.join(self.results_dir, "vio_results_*.csv"))
        csv_files.sort(reverse=True)  # Most recent first
        self.csv_files = csv_files
        return csv_files
    
    def display_file_browser(self):
        """Display interactive file browser."""
        clear_console()
        files = self.list_results_files()
        
        if not files:
            print(f"{Colors.FAIL}No results files found in {self.results_dir}{Colors.ENDC}")
            input("Press Enter to continue...")
            return None
        
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}")
        print(f"{'AVAILABLE RESULTS FILES':^80}")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        for i, file in enumerate(files, 1):
            file_stat = os.stat(file)
            file_size = file_stat.st_size / 1024  # KB
            mod_time = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            filename = os.path.basename(file)
            print(f"{Colors.OKCYAN}{i:2d}{Colors.ENDC}. {filename:<50} {file_size:>8.1f} KB  {mod_time}")
        
        print(f"\n{Colors.OKGREEN}Enter file number (1-{len(files)}) or 'q' to quit: {Colors.ENDC}", end="")
        choice = input().strip().lower()
        
        if choice == 'q':
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]
            else:
                print(f"{Colors.FAIL}Invalid selection!{Colors.ENDC}")
                input("Press Enter to continue...")
                return None
        except ValueError:
            print(f"{Colors.FAIL}Invalid input!{Colors.ENDC}")
            input("Press Enter to continue...")
            return None
    
    def load_csv(self, filepath: str) -> bool:
        """Load CSV file and parse parameters, estimated trajectory, and ground truth."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Parse parameters section
            self.parameters = {}
            for line in lines:
                if line.startswith('# ') and ':' in line:
                    line_clean = line[2:].strip()
                    if 'VIO PARAMETERS' not in line_clean and '====' not in line_clean and 'TRAJECTORY' not in line_clean:
                        key, value = line_clean.split(':', 1)
                        self.parameters[key.strip()] = value.strip()
        
            # Find section starts
            est_start = None
            gt_start = None
            
            for i, line in enumerate(lines):
                if line.startswith('est_timestamp,'):
                    est_start = i
                elif line.startswith('gt_timestamp,'):
                    gt_start = i
            
            if est_start is None or gt_start is None:
                raise ValueError("Could not find trajectory sections in CSV")
            
            # Parse estimated trajectory
            est_data = []
            for i in range(est_start + 1, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('#') or line.startswith('gt_timestamp'):
                    break
                parts = line.split(',')
                if len(parts) == 4:
                    est_data.append({
                        'timestamp': float(parts[0]),
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'z': float(parts[3])
                    })
        
            # Parse ground truth trajectory
            gt_data = []
            for i in range(gt_start + 1, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('#'):
                    break
                parts = line.split(',')
                if len(parts) == 4:
                    gt_data.append({
                        'timestamp': float(parts[0]),
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'z': float(parts[3])
                    })
        
            # Store raw data
            self.est_data = {
                'timestamp': np.array([d['timestamp'] for d in est_data]),
                'x': np.array([d['x'] for d in est_data]),
                'y': np.array([d['y'] for d in est_data]),
                'z': np.array([d['z'] for d in est_data])
            }
        
            self.gt_data = {
                'timestamp': np.array([d['timestamp'] for d in gt_data]),
                'x': np.array([d['x'] for d in gt_data]),
                'y': np.array([d['y'] for d in gt_data]),
                'z': np.array([d['z'] for d in gt_data])
            }
        
            # Associate and create aligned data for metrics
            self._associate_trajectories()
        
            self.current_file = filepath
            self.metrics_cache = {}
            clear_console()
            print(f"{Colors.OKGREEN}✓ Successfully loaded: {os.path.basename(filepath)}{Colors.ENDC}")
            print(f"  Estimated poses: {len(self.est_data['timestamp'])}")
            print(f"  Ground truth points: {len(self.gt_data['timestamp'])}")
            print(f"  Associated pairs: {len(self.data['timestamp'])}")
            input("Press Enter to continue...")
            return True
        except Exception as e:
            clear_console()
            print(f"{Colors.FAIL}✗ Error loading file: {e}{Colors.ENDC}")
            input("Press Enter to continue...")
            return False

    def _associate_trajectories(self, max_dt: float = 0.25):
        """Associate estimated poses with nearest ground truth by timestamp."""
        est_timestamps = self.est_data['timestamp']
        gt_timestamps = self.gt_data['timestamp']
        
        associated_est_x = []
        associated_est_y = []
        associated_est_z = []
        associated_gt_x = []
        associated_gt_y = []
        associated_gt_z = []
        associated_timestamps = []
        associated_dt = []
        
        for i, t_est in enumerate(est_timestamps):
            # Find closest GT timestamp
            dt_array = np.abs(gt_timestamps - t_est)
            min_idx = np.argmin(dt_array)
            min_dt = dt_array[min_idx]
            
            if min_dt <= max_dt:
                associated_timestamps.append(t_est)
                associated_est_x.append(self.est_data['x'][i])
                associated_est_y.append(self.est_data['y'][i])
                associated_est_z.append(self.est_data['z'][i])
                associated_gt_x.append(self.gt_data['x'][min_idx])
                associated_gt_y.append(self.gt_data['y'][min_idx])
                associated_gt_z.append(self.gt_data['z'][min_idx])
                associated_dt.append(min_dt)
        
        # Store associated data (compatible with existing code)
        self.data = {
            'timestamp': np.array(associated_timestamps),
            'est_x': np.array(associated_est_x),
            'est_y': np.array(associated_est_y),
            'est_z': np.array(associated_est_z),
            'gt_x': np.array(associated_gt_x),
            'gt_y': np.array(associated_gt_y),
            'gt_z': np.array(associated_gt_z),
            'dt': np.array(associated_dt)  # Time difference for each association
        }
    
    def compute_metrics(self) -> Dict:
        """Compute evaluation metrics."""
        if self.data is None:
            return {}
        
        if 'all' in self.metrics_cache:
            return self.metrics_cache['all']
        
        est_pos = np.column_stack([self.data['est_x'], self.data['est_y'], self.data['est_z']])
        gt_pos = np.column_stack([self.data['gt_x'], self.data['gt_y'], self.data['gt_z']])
        
        errors = np.linalg.norm(est_pos - gt_pos, axis=1)
        
        metrics = {
            'ape_rmse': np.sqrt(np.mean(errors ** 2)),
            'ape_max': np.max(errors),
            'ape_min': np.min(errors),
            'ape_mean': np.mean(errors),
            'ape_std': np.std(errors),
            'num_poses': len(errors),
            'duration': self.data['timestamp'][-1] - self.data['timestamp'][0],
            'errors': errors
        }
        
        self.metrics_cache['all'] = metrics
        return metrics
    
    def display_parameters(self):
      """Display loaded parameters organized by category in two columns."""
      clear_console()
      if not self.parameters:
          print("No parameters loaded.")
          input("Press Enter to continue...")
          return
      
      print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}")
      print(f"{'VIO PARAMETERS':^80}")
      print(f"{'='*80}{Colors.ENDC}\n")
      
      # Define parameter categories (matching vioParams.cfg structure)
      categories = {
          'General VIO': [
              'MAX_FEATURES', 'MIN_FEATURES', 'MIN_TRACKED_FEATURES',
              'MAX_TRACKING_ERROR_PX', 'MAX_TRACKING_AGE',
              'KF_FEATURE_THRESHOLD', 'KF_PARALLAX_THRESHOLD',
              'GPS_PRIOR_INTERVAL', 'GPS_PRIOR_ENABLE'
          ],
          'Feature Detection (GFTT)': [
              'GFTT_MAX_FEATURES', 'GFTT_QUALITY', 'GFTT_MIN_DIST', 'GFTT_BLOCK_SIZE'
          ],
          'Feature Tracking (KLT)': [
              'KLT_MAX_LEVEL', 'KLT_ITERS', 'KLT_EPS', 'KLT_MIN_EIG', 
              'KLT_FB_THRESH_PX', 'KLT_BORDER_MARGIN'
          ],
          'Information Rate': [
              'CAMERA_RATE', 'INERTIAL_RATE'
          ],
          'VO Noise (per-axis)': [
              'VO_NOISE_ROT_X', 'VO_NOISE_ROT_Y', 'VO_NOISE_ROT_Z',
              'VO_NOISE_TRANS_X', 'VO_NOISE_TRANS_Y', 'VO_NOISE_TRANS_Z'
          ],
          'Inertial Prior Noise (per-axis)': [
              'ROT_PRIOR_NOISE_X', 'ROT_PRIOR_NOISE_Y', 'ROT_PRIOR_NOISE_Z',
              'TRANS_PRIOR_NOISE_X', 'TRANS_PRIOR_NOISE_Y', 'TRANS_PRIOR_NOISE_Z'
          ],
          'GPS Noise (per-axis)': [
              'GPS_NOISE_X', 'GPS_NOISE_Y', 'GPS_NOISE_Z'
          ],
          'Altitude Prior Noise': [
              'ALT_PRIOR_NOISE'
          ]
      }
      
      for category, param_keys in categories.items():
          # Get parameters that exist in this category
          cat_params = [(k, self.parameters[k]) for k in param_keys if k in self.parameters]
          
          if not cat_params:
              continue
          
          print(f"{Colors.OKBLUE}{Colors.BOLD}┌─ {category} {'─' * (76 - len(category))}┐{Colors.ENDC}")
          
          # Display in two columns
          mid = (len(cat_params) + 1) // 2
          
          for i in range(mid):
              left_key, left_val = cat_params[i]
              left_str = f"│ {Colors.OKCYAN}{left_key:<28}{Colors.ENDC} {left_val:<10}"
              
              if i + mid < len(cat_params):
                  right_key, right_val = cat_params[i + mid]
                  right_str = f"{Colors.OKCYAN}{right_key:<28}{Colors.ENDC} {right_val:<10}│"
                  print(f"{left_str} │ {right_str}")
              else:
                  print(f"{left_str} │{' ' * 41}│")
          
          print(f"{Colors.OKBLUE}└{'─' * 78}┘{Colors.ENDC}")
          print()
      
      # Show any uncategorized parameters
      all_categorized = set()
      for keys in categories.values():
          all_categorized.update(keys)
      
      uncategorized = [(k, v) for k, v in self.parameters.items() if k not in all_categorized]
      
      if uncategorized:
          print(f"{Colors.WARNING}{Colors.BOLD}┌─ Other Parameters {'─' * 58}┐{Colors.ENDC}")
          for key, val in uncategorized:
              print(f"│ {Colors.OKCYAN}{key:<28}{Colors.ENDC} {val:<47}│")
          print(f"{Colors.WARNING}└{'─' * 78}┘{Colors.ENDC}")
          print()
      
      input("Press Enter to continue...")
    
    def display_global_metrics(self):
        """Display global metrics."""
        clear_console()
        metrics = self.compute_metrics()
        if not metrics:
            print(f"{Colors.FAIL}No data loaded.{Colors.ENDC}")
            input("Press Enter to continue...")
            return
        
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}")
        print(f"{'GLOBAL METRICS':^80}")
        print(f"{'='*80}{Colors.ENDC}\n")
        
        print(f"  {Colors.OKGREEN}APE RMSE{Colors.ENDC}     : {metrics['ape_rmse']:>10.4f} m")
        print(f"  {Colors.OKGREEN}APE MAX{Colors.ENDC}      : {metrics['ape_max']:>10.4f} m")
        print(f"  {Colors.OKGREEN}APE MIN{Colors.ENDC}      : {metrics['ape_min']:>10.4f} m")
        print(f"  {Colors.OKGREEN}APE MEAN{Colors.ENDC}     : {metrics['ape_mean']:>10.4f} m")
        print(f"  {Colors.OKGREEN}APE STD{Colors.ENDC}      : {metrics['ape_std']:>10.4f} m")
        print(f"  {Colors.OKGREEN}Total Poses{Colors.ENDC}  : {metrics['num_poses']:>10}")
        print(f"  {Colors.OKGREEN}Duration{Colors.ENDC}     : {metrics['duration']:>10.2f} seconds")
        
        # Association statistics
        if 'dt' in self.data:
            print(f"\n{Colors.OKBLUE}{Colors.BOLD}Association Statistics:{Colors.ENDC}")
            print(f"  {Colors.OKCYAN}Mean Δt{Colors.ENDC}      : {np.mean(self.data['dt'])*1000:>10.2f} ms")
            print(f"  {Colors.OKCYAN}Max Δt{Colors.ENDC}       : {np.max(self.data['dt'])*1000:>10.2f} ms")
            print(f"  {Colors.OKCYAN}Min Δt{Colors.ENDC}       : {np.min(self.data['dt'])*1000:>10.2f} ms")
        
        print()
        input("Press Enter to continue...")
    
    def interactive_trajectory_viewer(self):
        """Launch interactive GUI trajectory viewer."""
        if self.data is None:
            clear_console()
            print(f"{Colors.FAIL}No data loaded.{Colors.ENDC}")
            input("Press Enter to continue...")
            return
        
        metrics = self.compute_metrics()
        viewer = InteractiveTrajectoryViewer(self.data, metrics)
        viewer.show()
    
    def plot_trajectory_3d(self):
        """Plot 3D trajectory comparison."""
        if self.data is None:
            clear_console()
            print(f"{Colors.FAIL}No data loaded.{Colors.ENDC}")
            input("Press Enter to continue...")
            return
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D trajectory
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(self.data['est_x'], self.data['est_y'], self.data['est_z'], 
                'b-', label='Estimated', linewidth=2, alpha=0.7)
        ax1.plot(self.data['gt_x'], self.data['gt_y'], self.data['gt_z'], 
                'r--', label='Ground Truth', linewidth=2, alpha=0.7)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Error over time
        metrics = self.compute_metrics()
        ax2 = fig.add_subplot(122)
        ax2.plot(self.data['timestamp'] - self.data['timestamp'][0], 
                metrics['errors'], 'g-', linewidth=2)
        ax2.axhline(y=metrics['ape_rmse'], color='b', linestyle='--', 
                   label=f"RMSE: {metrics['ape_rmse']:.4f} m")
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Absolute Position Error (m)')
        ax2.set_title('Position Error Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_xyz_comparison(self):
        """Plot X, Y, Z coordinate comparisons."""
        if self.data is None:
            clear_console()
            print(f"{Colors.FAIL}No data loaded.{Colors.ENDC}")
            input("Press Enter to continue...")
            return
        
        time_rel = self.data['timestamp'] - self.data['timestamp'][0]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        coords = [('X', 'est_x', 'gt_x'), ('Y', 'est_y', 'gt_y'), ('Z', 'est_z', 'gt_z')]
        
        for idx, (label, est_key, gt_key) in enumerate(coords):
            axes[idx].plot(time_rel, self.data[est_key], 'b-', label='Estimated', linewidth=2)
            axes[idx].plot(time_rel, self.data[gt_key], 'r--', label='Ground Truth', linewidth=2)
            axes[idx].set_ylabel(f'{label} (m)')
            axes[idx].legend()
            axes[idx].grid(True)
        
        axes[-1].set_xlabel('Time (s)')
        plt.suptitle('Coordinate Comparison Over Time')
        plt.tight_layout()
        plt.show()
    
    def run_interactive_session(self):
        """Run interactive evaluation session."""
        while True:
            clear_console()
            print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}")
            print(f"{'VIO RESULTS EVALUATOR':^80}")
            print(f"{'='*80}{Colors.ENDC}\n")
            
            current_file = os.path.basename(self.current_file) if self.current_file else "None"
            print(f"Current File: {Colors.OKCYAN}{current_file}{Colors.ENDC}\n")
            
            print(f"{Colors.OKGREEN}Main Menu:{Colors.ENDC}")
            print(f"  {Colors.OKCYAN}1{Colors.ENDC} - Load results file")
            print(f"  {Colors.OKCYAN}2{Colors.ENDC} - Show parameters")
            print(f"  {Colors.OKCYAN}3{Colors.ENDC} - Show global metrics")
            print(f"  {Colors.OKCYAN}4{Colors.ENDC} - Interactive trajectory viewer (GUI)")
            print(f"  {Colors.OKCYAN}5{Colors.ENDC} - Plot 3D trajectory")
            print(f"  {Colors.OKCYAN}6{Colors.ENDC} - Plot XYZ comparison")
            print(f"  {Colors.OKCYAN}q{Colors.ENDC} - Quit")
            
            print(f"\n{Colors.OKGREEN}Enter choice: {Colors.ENDC}", end="")
            choice = input().strip().lower()
            
            if choice == '1':
                filepath = self.display_file_browser()
                if filepath:
                    self.load_csv(filepath)
            elif choice == '2':
                self.display_parameters()
            elif choice == '3':
                self.display_global_metrics()
            elif choice == '4':
                self.interactive_trajectory_viewer()
            elif choice == '5':
                self.plot_trajectory_3d()
            elif choice == '6':
                self.plot_xyz_comparison()
            elif choice == 'q':
                clear_console()
                print(f"{Colors.OKGREEN}Goodbye!{Colors.ENDC}")
                break
            else:
                clear_console()
                print(f"{Colors.FAIL}Invalid choice!{Colors.ENDC}")
                input("Press Enter to continue...")

def main():
    parser = argparse.ArgumentParser(description="VIO Results Offline Evaluator")
    parser.add_argument('--results-dir', 
                       default="/home/tony/Desktop/MEEC-SA/2º Ano/FUIMS/Resultados",
                       help='Path to results directory')
    args = parser.parse_args()
    
    evaluator = VIOResultsEvaluator(args.results_dir)
    evaluator.run_interactive_session()

if __name__ == '__main__':
    main()