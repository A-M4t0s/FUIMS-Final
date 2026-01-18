#!/usr/bin/env python3
"""
VIO Results Comparison GUI
Interactive tool for comparing multiple VIO results from consolidated CSV files.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

# Default colors for trajectory comparison
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
]

@dataclass
class TestResult:
    """Data class for a single test result."""
    filename: str
    parameters: Dict[str, str]
    metrics: Dict[str, float]
    est_trajectory: Dict[str, np.ndarray]

@dataclass
class ConsolidatedData:
    """Data class for consolidated CSV data."""
    results: List[TestResult]
    gt_trajectory: Dict[str, np.ndarray]
    param_keys: List[str]
    metric_keys: List[str]


class ConsolidatedCSVParser:
    """Parser for consolidated VIO results CSV files."""
    
    # Key parameters to highlight
    KEY_PARAMS = [
        'GPS_PRIOR_ENABLE', 'GPS_PRIOR_INTERVAL',
        'VO_NOISE_ROT_X', 'VO_NOISE_ROT_Y', 'VO_NOISE_ROT_Z',
        'VO_NOISE_TRANS_X', 'VO_NOISE_TRANS_Y', 'VO_NOISE_TRANS_Z',
        'ROT_PRIOR_NOISE_X', 'ROT_PRIOR_NOISE_Y', 'ROT_PRIOR_NOISE_Z',
        'TRANS_PRIOR_NOISE_X', 'TRANS_PRIOR_NOISE_Y', 'TRANS_PRIOR_NOISE_Z',
        'GPS_NOISE_X', 'GPS_NOISE_Y', 'GPS_NOISE_Z',
        'ALT_PRIOR_NOISE'
    ]
    
    METRIC_KEYS = ['ape_rmse', 'ape_max', 'ape_min', 'ape_mean', 'ape_std', 'num_poses', 'duration']
    
    @staticmethod
    def parse(filepath: str) -> Optional[ConsolidatedData]:
        """Parse a consolidated CSV file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            results = []
            gt_trajectory = None
            param_keys = []
            current_section = None
            current_est_file = None
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Detect section headers
                if '# ===== PARAMETERS AND METRICS =====' in line:
                    current_section = 'params'
                    i += 1
                    continue
                elif '# ===== GROUND TRUTH TRAJECTORY =====' in line:
                    current_section = 'gt'
                    i += 1
                    continue
                elif '# ===== ESTIMATED TRAJECTORY:' in line:
                    current_section = 'est'
                    # Extract filename from header
                    current_est_file = line.split(':')[1].strip().replace('=', '').strip()
                    i += 1
                    continue
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    i += 1
                    continue
                
                # Parse parameters and metrics section
                if current_section == 'params':
                    if line.startswith('filename,'):
                        # Header row
                        header = line.split(',')
                        param_keys = [h for h in header[1:] if h not in ConsolidatedCSVParser.METRIC_KEYS]
                    else:
                        # Data row
                        parts = line.split(',')
                        if len(parts) > 1:
                            filename = parts[0]
                            values = parts[1:]
                            
                            # Parse parameters
                            params = {}
                            metrics = {}
                            
                            for j, key in enumerate(param_keys):
                                if j < len(values):
                                    params[key] = values[j]
                            
                            # Parse metrics (they come after parameters)
                            metric_start = len(param_keys)
                            for j, key in enumerate(ConsolidatedCSVParser.METRIC_KEYS):
                                idx = metric_start + j
                                if idx < len(values):
                                    try:
                                        metrics[key] = float(values[idx])
                                    except ValueError:
                                        metrics[key] = 0.0
                            
                            results.append(TestResult(
                                filename=filename,
                                parameters=params,
                                metrics=metrics,
                                est_trajectory={'timestamp': [], 'x': [], 'y': [], 'z': []}
                            ))
                
                # Parse ground truth trajectory
                elif current_section == 'gt':
                    if line.startswith('gt_timestamp,'):
                        gt_trajectory = {'timestamp': [], 'x': [], 'y': [], 'z': []}
                    else:
                        parts = line.split(',')
                        if len(parts) == 4 and gt_trajectory is not None:
                            try:
                                gt_trajectory['timestamp'].append(float(parts[0]))
                                gt_trajectory['x'].append(float(parts[1]))
                                gt_trajectory['y'].append(float(parts[2]))
                                gt_trajectory['z'].append(float(parts[3]))
                            except ValueError:
                                pass
                
                # Parse estimated trajectory
                elif current_section == 'est':
                    if line.startswith('est_timestamp,'):
                        pass  # Skip header
                    else:
                        parts = line.split(',')
                        if len(parts) == 4 and current_est_file:
                            # Find the matching result
                            for result in results:
                                if result.filename == current_est_file:
                                    try:
                                        result.est_trajectory['timestamp'].append(float(parts[0]))
                                        result.est_trajectory['x'].append(float(parts[1]))
                                        result.est_trajectory['y'].append(float(parts[2]))
                                        result.est_trajectory['z'].append(float(parts[3]))
                                    except ValueError:
                                        pass
                                    break
                
                i += 1
            
            # Convert lists to numpy arrays
            if gt_trajectory:
                gt_trajectory = {
                    'timestamp': np.array(gt_trajectory['timestamp']),
                    'x': np.array(gt_trajectory['x']),
                    'y': np.array(gt_trajectory['y']),
                    'z': np.array(gt_trajectory['z'])
                }
            
            for result in results:
                result.est_trajectory = {
                    'timestamp': np.array(result.est_trajectory['timestamp']),
                    'x': np.array(result.est_trajectory['x']),
                    'y': np.array(result.est_trajectory['y']),
                    'z': np.array(result.est_trajectory['z'])
                }
            
            return ConsolidatedData(
                results=results,
                gt_trajectory=gt_trajectory,
                param_keys=param_keys,
                metric_keys=ConsolidatedCSVParser.METRIC_KEYS
            )
            
        except Exception as e:
            print(f"Error parsing CSV: {e}")
            import traceback
            traceback.print_exc()
            return None


class VIOComparisonGUI:
    """Main GUI application for VIO results comparison."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VIO Results Comparison Tool")
        self.root.geometry("1600x900")
        
        # Data
        self.data: Optional[ConsolidatedData] = None
        self.selected_indices: List[int] = []
        
        # Setup UI
        self._setup_menu()
        self._setup_main_layout()
        self._setup_styles()
        
    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.configure("Treeview", rowheight=25)
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
        
    def _setup_menu(self):
        """Setup menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open CSV...", command=self._open_file, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._open_file())
        
    def _setup_main_layout(self):
        """Setup main window layout."""
        # Main paned window (horizontal split)
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Results table and parameters
        self.left_frame = ttk.Frame(self.main_paned, width=500)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Right panel - Plots
        self.right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_frame, weight=2)
        
        self._setup_left_panel()
        self._setup_right_panel()
        
    def _setup_left_panel(self):
        """Setup left panel with results table and parameters."""
        # Top section - Results table
        table_frame = ttk.LabelFrame(self.left_frame, text="Test Results (Select to Compare)")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results treeview
        columns = ('filename', 'rmse', 'max_err', 'mean_err', 'gps_enable', 'gps_interval')
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings', selectmode='extended')
        
        self.results_tree.heading('filename', text='Filename')
        self.results_tree.heading('rmse', text='RMSE (m)')
        self.results_tree.heading('max_err', text='Max (m)')
        self.results_tree.heading('mean_err', text='Mean (m)')
        self.results_tree.heading('gps_enable', text='GPS')
        self.results_tree.heading('gps_interval', text='GPS Int.')
        
        self.results_tree.column('filename', width=180, minwidth=100)
        self.results_tree.column('rmse', width=80, minwidth=60, anchor='center')
        self.results_tree.column('max_err', width=80, minwidth=60, anchor='center')
        self.results_tree.column('mean_err', width=80, minwidth=60, anchor='center')
        self.results_tree.column('gps_enable', width=50, minwidth=40, anchor='center')
        self.results_tree.column('gps_interval', width=70, minwidth=50, anchor='center')
        
        # Scrollbars
        y_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        self.results_tree.grid(row=0, column=0, sticky='nsew')
        y_scroll.grid(row=0, column=1, sticky='ns')
        x_scroll.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection event
        self.results_tree.bind('<<TreeviewSelect>>', self._on_selection_changed)
        
        # Buttons frame
        btn_frame = ttk.Frame(self.left_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Compare Selected", command=self._compare_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Select All", command=self._select_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear Selection", command=self._clear_selection).pack(side=tk.LEFT, padx=2)
        
        # Bottom section - Key Parameters display
        params_frame = ttk.LabelFrame(self.left_frame, text="Key Parameters (Selected)")
        params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Parameters text widget
        self.params_text = tk.Text(params_frame, height=15, font=('Courier', 9), wrap=tk.NONE)
        params_scroll_y = ttk.Scrollbar(params_frame, orient=tk.VERTICAL, command=self.params_text.yview)
        params_scroll_x = ttk.Scrollbar(params_frame, orient=tk.HORIZONTAL, command=self.params_text.xview)
        self.params_text.configure(yscrollcommand=params_scroll_y.set, xscrollcommand=params_scroll_x.set)
        
        self.params_text.grid(row=0, column=0, sticky='nsew')
        params_scroll_y.grid(row=0, column=1, sticky='ns')
        params_scroll_x.grid(row=1, column=0, sticky='ew')
        
        params_frame.grid_rowconfigure(0, weight=1)
        params_frame.grid_columnconfigure(0, weight=1)
        
        # Configure text tags for coloring
        self.params_text.tag_configure('header', foreground='#1f77b4', font=('Courier', 9, 'bold'))
        self.params_text.tag_configure('param_name', foreground='#2ca02c')
        self.params_text.tag_configure('value', foreground='#d62728')
        
    def _setup_right_panel(self):
        """Setup right panel with plots."""
        # Notebook for different plot views
        self.plot_notebook = ttk.Notebook(self.right_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: 3D Trajectory comparison
        self.traj_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.traj_frame, text="3D Trajectory")
        
        # Tab 2: 2D Projections
        self.proj_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.proj_frame, text="2D Projections")
        
        # Tab 3: Error comparison
        self.error_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.error_frame, text="Error Analysis")
        
        # Tab 4: Metrics comparison bar chart
        self.metrics_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.metrics_frame, text="Metrics Comparison")
        
        # Setup figures for each tab
        self._setup_trajectory_plot()
        self._setup_projection_plot()
        self._setup_error_plot()
        self._setup_metrics_plot()
        
    def _setup_trajectory_plot(self):
        """Setup 3D trajectory plot."""
        self.traj_fig = Figure(figsize=(8, 6), dpi=100)
        self.traj_ax = self.traj_fig.add_subplot(111, projection='3d')
        self.traj_ax.set_xlabel('X (m)')
        self.traj_ax.set_ylabel('Y (m)')
        self.traj_ax.set_zlabel('Z (m)')
        self.traj_ax.set_title('3D Trajectory Comparison')
        
        self.traj_canvas = FigureCanvasTkAgg(self.traj_fig, master=self.traj_frame)
        self.traj_canvas.draw()
        
        toolbar = NavigationToolbar2Tk(self.traj_canvas, self.traj_frame)
        toolbar.update()
        
        self.traj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _setup_projection_plot(self):
        """Setup 2D projection plots."""
        self.proj_fig = Figure(figsize=(8, 6), dpi=100)
        self.proj_axes = {
            'xy': self.proj_fig.add_subplot(131),
            'xz': self.proj_fig.add_subplot(132),
            'yz': self.proj_fig.add_subplot(133)
        }
        
        self.proj_axes['xy'].set_xlabel('X (m)')
        self.proj_axes['xy'].set_ylabel('Y (m)')
        self.proj_axes['xy'].set_title('XY Plane')
        
        self.proj_axes['xz'].set_xlabel('X (m)')
        self.proj_axes['xz'].set_ylabel('Z (m)')
        self.proj_axes['xz'].set_title('XZ Plane')
        
        self.proj_axes['yz'].set_xlabel('Y (m)')
        self.proj_axes['yz'].set_ylabel('Z (m)')
        self.proj_axes['yz'].set_title('YZ Plane')
        
        self.proj_fig.tight_layout()
        
        self.proj_canvas = FigureCanvasTkAgg(self.proj_fig, master=self.proj_frame)
        self.proj_canvas.draw()
        
        toolbar = NavigationToolbar2Tk(self.proj_canvas, self.proj_frame)
        toolbar.update()
        
        self.proj_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _setup_error_plot(self):
        """Setup error analysis plots."""
        self.error_fig = Figure(figsize=(8, 6), dpi=100)
        self.error_axes = {
            'time': self.error_fig.add_subplot(211),
            'hist': self.error_fig.add_subplot(212)
        }
        
        self.error_axes['time'].set_xlabel('Time (s)')
        self.error_axes['time'].set_ylabel('Position Error (m)')
        self.error_axes['time'].set_title('Error Over Time')
        self.error_axes['time'].grid(True, alpha=0.3)
        
        self.error_axes['hist'].set_xlabel('Position Error (m)')
        self.error_axes['hist'].set_ylabel('Frequency')
        self.error_axes['hist'].set_title('Error Distribution')
        self.error_axes['hist'].grid(True, alpha=0.3)
        
        self.error_fig.tight_layout()
        
        self.error_canvas = FigureCanvasTkAgg(self.error_fig, master=self.error_frame)
        self.error_canvas.draw()
        
        toolbar = NavigationToolbar2Tk(self.error_canvas, self.error_frame)
        toolbar.update()
        
        self.error_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _setup_metrics_plot(self):
        """Setup metrics comparison bar chart."""
        self.metrics_fig = Figure(figsize=(8, 6), dpi=100)
        self.metrics_ax = self.metrics_fig.add_subplot(111)
        self.metrics_ax.set_title('Metrics Comparison')
        
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, master=self.metrics_frame)
        self.metrics_canvas.draw()
        
        toolbar = NavigationToolbar2Tk(self.metrics_canvas, self.metrics_frame)
        toolbar.update()
        
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _open_file(self):
        """Open a consolidated CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select Consolidated CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.expanduser("~/Desktop")
        )
        
        if not filepath:
            return
        
        self.data = ConsolidatedCSVParser.parse(filepath)
        
        if self.data is None:
            messagebox.showerror("Error", "Failed to parse CSV file.")
            return
        
        self._populate_results_table()
        self._clear_plots()
        
        self.root.title(f"VIO Results Comparison - {os.path.basename(filepath)}")
        messagebox.showinfo("Success", f"Loaded {len(self.data.results)} test results.")
        
    def _populate_results_table(self):
        """Populate the results treeview."""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if self.data is None:
            return
        
        # Add results sorted by RMSE
        sorted_results = sorted(enumerate(self.data.results), 
                               key=lambda x: x[1].metrics.get('ape_rmse', float('inf')))
        
        for idx, result in sorted_results:
            gps_enable = result.parameters.get('GPS_PRIOR_ENABLE', 'N/A')
            gps_interval = result.parameters.get('GPS_PRIOR_INTERVAL', 'N/A')
            
            self.results_tree.insert('', 'end', iid=str(idx), values=(
                result.filename,
                f"{result.metrics.get('ape_rmse', 0):.4f}",
                f"{result.metrics.get('ape_max', 0):.4f}",
                f"{result.metrics.get('ape_mean', 0):.4f}",
                gps_enable,
                gps_interval
            ))
            
    def _on_selection_changed(self, event):
        """Handle selection change in results table."""
        selected = self.results_tree.selection()
        self.selected_indices = [int(iid) for iid in selected]
        self._update_params_display()
        
    def _update_params_display(self):
        """Update the key parameters display."""
        self.params_text.config(state=tk.NORMAL)
        self.params_text.delete(1.0, tk.END)
        
        if not self.data or not self.selected_indices:
            self.params_text.insert(tk.END, "Select test results to view parameters...")
            self.params_text.config(state=tk.DISABLED)
            return
        
        # Key parameters to display
        key_params = ConsolidatedCSVParser.KEY_PARAMS
        
        # Build comparison table
        header = f"{'Parameter':<25}"
        for idx in self.selected_indices:
            result = self.data.results[idx]
            short_name = result.filename[:15] + "..." if len(result.filename) > 15 else result.filename
            header += f" │ {short_name:^15}"
        
        self.params_text.insert(tk.END, "=" * len(header) + "\n")
        self.params_text.insert(tk.END, "KEY PARAMETERS COMPARISON\n", 'header')
        self.params_text.insert(tk.END, "=" * len(header) + "\n\n")
        
        self.params_text.insert(tk.END, header + "\n")
        self.params_text.insert(tk.END, "-" * len(header) + "\n")
        
        for param in key_params:
            line = f"{param:<25}"
            for idx in self.selected_indices:
                result = self.data.results[idx]
                value = result.parameters.get(param, 'N/A')
                line += f" │ {value:^15}"
            self.params_text.insert(tk.END, line + "\n")
        
        # Add metrics section
        self.params_text.insert(tk.END, "\n" + "=" * len(header) + "\n")
        self.params_text.insert(tk.END, "METRICS COMPARISON\n", 'header')
        self.params_text.insert(tk.END, "=" * len(header) + "\n\n")
        
        metrics_to_show = ['ape_rmse', 'ape_max', 'ape_min', 'ape_mean', 'ape_std', 'num_poses', 'duration']
        metric_labels = {
            'ape_rmse': 'RMSE (m)',
            'ape_max': 'Max Error (m)',
            'ape_min': 'Min Error (m)',
            'ape_mean': 'Mean Error (m)',
            'ape_std': 'Std Dev (m)',
            'num_poses': 'Num Poses',
            'duration': 'Duration (s)'
        }
        
        for metric in metrics_to_show:
            line = f"{metric_labels.get(metric, metric):<25}"
            for idx in self.selected_indices:
                result = self.data.results[idx]
                value = result.metrics.get(metric, 0)
                if metric in ['num_poses']:
                    line += f" │ {int(value):^15}"
                else:
                    line += f" │ {value:^15.4f}"
            self.params_text.insert(tk.END, line + "\n")
        
        self.params_text.config(state=tk.DISABLED)
        
    def _compare_selected(self):
        """Compare selected results."""
        if not self.data or not self.selected_indices:
            messagebox.showwarning("Warning", "Please select at least one result to compare.")
            return
        
        self._update_trajectory_plot()
        self._update_projection_plot()
        self._update_error_plot()
        self._update_metrics_plot()
        
    def _clear_plots(self):
        """Clear all plots."""
        self.traj_ax.clear()
        self.traj_ax.set_xlabel('X (m)')
        self.traj_ax.set_ylabel('Y (m)')
        self.traj_ax.set_zlabel('Z (m)')
        self.traj_ax.set_title('3D Trajectory Comparison')
        self.traj_canvas.draw()
        
        for ax in self.proj_axes.values():
            ax.clear()
        self.proj_canvas.draw()
        
        for ax in self.error_axes.values():
            ax.clear()
        self.error_canvas.draw()
        
        self.metrics_ax.clear()
        self.metrics_canvas.draw()
        
    def _update_trajectory_plot(self):
        """Update 3D trajectory plot."""
        self.traj_ax.clear()
        
        if self.data is None:
            return
        
        gt = self.data.gt_trajectory
        
        # Plot ground truth
        self.traj_ax.plot(gt['x'], gt['y'], gt['z'], 
                         'k--', linewidth=2.5, label='Ground Truth', alpha=0.8)
        
        # Plot selected estimates
        for i, idx in enumerate(self.selected_indices):
            result = self.data.results[idx]
            est = result.est_trajectory
            color = COLORS[i % len(COLORS)]
            
            short_name = result.filename[:20] + "..." if len(result.filename) > 20 else result.filename
            self.traj_ax.plot(est['x'], est['y'], est['z'], 
                            color=color, linewidth=1.5, label=short_name, alpha=0.7)
        
        self.traj_ax.set_xlabel('X (m)')
        self.traj_ax.set_ylabel('Y (m)')
        self.traj_ax.set_zlabel('Z (m)')
        self.traj_ax.set_title('3D Trajectory Comparison')
        self.traj_ax.legend(loc='upper left', fontsize=8)
        self.traj_ax.grid(True, alpha=0.3)
        
        self.traj_canvas.draw()
        
    def _update_projection_plot(self):
        """Update 2D projection plots."""
        for ax in self.proj_axes.values():
            ax.clear()
        
        if self.data is None:
            return
        
        gt = self.data.gt_trajectory
        
        # Plot ground truth on all projections
        self.proj_axes['xy'].plot(gt['x'], gt['y'], 'k--', linewidth=2.5, label='Ground Truth', alpha=0.8)
        self.proj_axes['xz'].plot(gt['x'], gt['z'], 'k--', linewidth=2.5, label='Ground Truth', alpha=0.8)
        self.proj_axes['yz'].plot(gt['y'], gt['z'], 'k--', linewidth=2.5, label='Ground Truth', alpha=0.8)
        
        # Plot selected estimates
        for i, idx in enumerate(self.selected_indices):
            result = self.data.results[idx]
            est = result.est_trajectory
            color = COLORS[i % len(COLORS)]
            
            short_name = result.filename[:15] + "..." if len(result.filename) > 15 else result.filename
            self.proj_axes['xy'].plot(est['x'], est['y'], color=color, linewidth=1.5, label=short_name, alpha=0.7)
            self.proj_axes['xz'].plot(est['x'], est['z'], color=color, linewidth=1.5, alpha=0.7)
            self.proj_axes['yz'].plot(est['y'], est['z'], color=color, linewidth=1.5, alpha=0.7)
        
        self.proj_axes['xy'].set_xlabel('X (m)')
        self.proj_axes['xy'].set_ylabel('Y (m)')
        self.proj_axes['xy'].set_title('XY Plane')
        self.proj_axes['xy'].legend(loc='upper left', fontsize=7)
        self.proj_axes['xy'].grid(True, alpha=0.3)
        self.proj_axes['xy'].set_aspect('equal', 'datalim')
        
        self.proj_axes['xz'].set_xlabel('X (m)')
        self.proj_axes['xz'].set_ylabel('Z (m)')
        self.proj_axes['xz'].set_title('XZ Plane')
        self.proj_axes['xz'].grid(True, alpha=0.3)
        
        self.proj_axes['yz'].set_xlabel('Y (m)')
        self.proj_axes['yz'].set_ylabel('Z (m)')
        self.proj_axes['yz'].set_title('YZ Plane')
        self.proj_axes['yz'].grid(True, alpha=0.3)
        
        self.proj_fig.tight_layout()
        self.proj_canvas.draw()
        
    def _compute_errors(self, result: TestResult) -> tuple:
        """Compute position errors for a result against ground truth."""
        if self.data is None:
            return np.array([]), np.array([])
        
        gt = self.data.gt_trajectory
        est = result.est_trajectory
        
        # Associate by timestamp
        errors = []
        times = []
        max_dt = 0.25
        
        for i, t_est in enumerate(est['timestamp']):
            dt_array = np.abs(gt['timestamp'] - t_est)
            min_idx = np.argmin(dt_array)
            
            if dt_array[min_idx] <= max_dt:
                est_pos = np.array([est['x'][i], est['y'][i], est['z'][i]])
                gt_pos = np.array([gt['x'][min_idx], gt['y'][min_idx], gt['z'][min_idx]])
                error = np.linalg.norm(est_pos - gt_pos)
                errors.append(error)
                times.append(t_est - est['timestamp'][0])
        
        return np.array(times), np.array(errors)
        
    def _update_error_plot(self):
        """Update error analysis plots."""
        for ax in self.error_axes.values():
            ax.clear()
        
        if self.data is None or not self.selected_indices:
            return
        
        all_errors = []
        
        # Plot error over time for each selected result
        for i, idx in enumerate(self.selected_indices):
            result = self.data.results[idx]
            times, errors = self._compute_errors(result)
            color = COLORS[i % len(COLORS)]
            
            if len(errors) > 0:
                all_errors.append(errors)
                short_name = result.filename[:20] + "..." if len(result.filename) > 20 else result.filename
                self.error_axes['time'].plot(times, errors, color=color, linewidth=1.5, 
                                            label=f"{short_name} (RMSE: {result.metrics.get('ape_rmse', 0):.3f}m)", 
                                            alpha=0.7)
        
        self.error_axes['time'].set_xlabel('Time (s)')
        self.error_axes['time'].set_ylabel('Position Error (m)')
        self.error_axes['time'].set_title('Error Over Time')
        self.error_axes['time'].legend(loc='upper left', fontsize=7)
        self.error_axes['time'].grid(True, alpha=0.3)
        
        # Histogram of errors
        if all_errors:
            for i, (idx, errors) in enumerate(zip(self.selected_indices, all_errors)):
                result = self.data.results[idx]
                color = COLORS[i % len(COLORS)]
                short_name = result.filename[:15] + "..." if len(result.filename) > 15 else result.filename
                self.error_axes['hist'].hist(errors, bins=50, alpha=0.5, color=color, 
                                            label=short_name, density=True)
        
        self.error_axes['hist'].set_xlabel('Position Error (m)')
        self.error_axes['hist'].set_ylabel('Density')
        self.error_axes['hist'].set_title('Error Distribution')
        self.error_axes['hist'].legend(loc='upper right', fontsize=7)
        self.error_axes['hist'].grid(True, alpha=0.3)
        
        self.error_fig.tight_layout()
        self.error_canvas.draw()
        
    def _update_metrics_plot(self):
        """Update metrics comparison bar chart."""
        self.metrics_ax.clear()
        
        if self.data is None or not self.selected_indices:
            return
        
        metrics_to_plot = ['ape_rmse', 'ape_max', 'ape_mean', 'ape_std']
        metric_labels = ['RMSE', 'Max', 'Mean', 'Std Dev']
        
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / max(len(self.selected_indices), 1)
        
        for i, idx in enumerate(self.selected_indices):
            result = self.data.results[idx]
            values = [result.metrics.get(m, 0) for m in metrics_to_plot]
            color = COLORS[i % len(COLORS)]
            
            short_name = result.filename[:15] + "..." if len(result.filename) > 15 else result.filename
            offset = width * i - width * len(self.selected_indices) / 2 + width / 2
            bars = self.metrics_ax.bar(x + offset, values, width, label=short_name, color=color, alpha=0.8)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                self.metrics_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
        
        self.metrics_ax.set_ylabel('Error (m)')
        self.metrics_ax.set_title('Metrics Comparison')
        self.metrics_ax.set_xticks(x)
        self.metrics_ax.set_xticklabels(metric_labels)
        self.metrics_ax.legend(loc='upper right', fontsize=7)
        self.metrics_ax.grid(True, alpha=0.3, axis='y')
        
        self.metrics_fig.tight_layout()
        self.metrics_canvas.draw()
        
    def _select_all(self):
        """Select all results."""
        for item in self.results_tree.get_children():
            self.results_tree.selection_add(item)
        
    def _clear_selection(self):
        """Clear all selections."""
        self.results_tree.selection_remove(*self.results_tree.selection())
        self.selected_indices = []
        self._update_params_display()


def main():
    root = tk.Tk()
    app = VIOComparisonGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
