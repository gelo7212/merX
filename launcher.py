#!/usr/bin/env python3
"""
Comprehensive Test and Visualization Launcher for merX

This script provides a unified interface to run search accuracy tests
and launch the graphical database viewer for the merX memory system.
"""

import os
import sys
import argparse
import logging
import subprocess
import webbrowser
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required dependencies."""
    logger.info("Installing required dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def run_search_accuracy_test(args):
    """Run the search accuracy test with tree-of-trees structure."""
    logger.info("Starting Search Accuracy Test...")
    
    try:
        # Import and run the test
        from examples.search_accuracy_test import main as run_accuracy_test
        run_accuracy_test()
          # Open results if available
        results_file = "data/search_accuracy_test/search_accuracy_results.json"
        if os.path.exists(results_file):
            logger.info(f"Test results saved to: {results_file}")
            if args.open_results:
                abs_results_file = os.path.abspath(results_file)
                try:
                    if sys.platform.startswith('win'):
                        os.startfile(abs_results_file)
                    elif sys.platform.startswith('darwin'):
                        subprocess.call(['open', abs_results_file])
                    else:
                        subprocess.call(['xdg-open', abs_results_file])
                except Exception as e:
                    logger.warning(f"Could not open results file: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Search accuracy test failed: {e}")
        return False

def launch_database_viewer(args):
    """Launch the graphical database viewer."""
    logger.info("Launching Database Viewer...")
    try:        
        if args.streamlit:
            # Launch Streamlit dashboard
            logger.info("Starting Streamlit dashboard...")
            subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "examples/streamlit_dashboard.py",
                "--server.port", "8501",
                "--server.headless", "true"
            ])
            
            # Open browser
            time.sleep(3)  # Wait for server to start
            webbrowser.open("http://localhost:8501")
            logger.info("Streamlit dashboard available at: http://localhost:8501")
            
        else:
            # Run standalone enhanced 3D viewer
            from examples.enhanced_3d_viewer import main as run_enhanced_viewer
            run_enhanced_viewer()
            logger.info("Enhanced 3D visualizations created in data/visualizations/")
              # Open the enhanced 3D visualization
            viz_files = [
                "data/visualizations/memory_3d_network.html",
                "data/visualizations/memory_tree_forest_3d.html", 
                "data/visualizations/memory_fractal_hierarchy.html",
                "data/visualizations/memory_dashboard.html"
            ]
            
            for viz_file in viz_files:
                if os.path.exists(viz_file) and args.open_results:
                    webbrowser.open(f"file://{os.path.abspath(viz_file)}")
                    break  # Open first available file
        
        return True
        
    except Exception as e:
        logger.error(f"Database viewer failed: {e}")
        return False

def run_extreme_performance_test(args):
    """Run the extreme performance test with 100,000+ nodes."""
    logger.info("Starting Extreme Performance Test...")
    
    try:
        # Import and run the test
        from examples.extreme_performance_test import main as run_extreme_test
        run_extreme_test()
        
        # Open results if available
        results_dir = "data/test_output"
        metrics_file = f"{results_dir}/metrics.json"
        
        if os.path.exists(metrics_file):
            logger.info(f"Test metrics saved to: {metrics_file}")
            if args.open_results:
                abs_metrics_file = os.path.abspath(metrics_file)
                try:
                    if sys.platform.startswith('win'):
                        os.startfile(abs_metrics_file)
                    elif sys.platform.startswith('darwin'):
                        subprocess.call(['open', abs_metrics_file])
                    else:
                        subprocess.call(['xdg-open', abs_metrics_file])
                except Exception as e:
                    logger.warning(f"Could not open metrics file: {e}")
        
        # Also check for log file
        log_file = "extreme_test_results.log"
        if os.path.exists(log_file):
            logger.info(f"Test logs saved to: {log_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Extreme performance test failed: {e}")
        return False

def run_tree_100_test(args):
    """Run the tree-of-trees test with exactly 100 nodes."""
    logger.info("Starting Tree-100 Test...")
    
    try:
        # Import and run the test
        from examples.tree_100_test import main as run_tree_test
        success = run_tree_test()
        
        if success:
            # Open results if available
            results_file = "data/search_accuracy_test/tree_100_results.json"
            if os.path.exists(results_file):
                logger.info(f"Test results saved to: {results_file}")
                if args.open_results:
                    abs_results_file = os.path.abspath(results_file)
                    try:
                        if sys.platform.startswith('win'):
                            os.startfile(abs_results_file)
                        elif sys.platform.startswith('darwin'):
                            subprocess.call(['open', abs_results_file])
                        else:
                            subprocess.call(['xdg-open', abs_results_file])
                    except Exception as e:
                        logger.warning(f"Could not open results file: {e}")
        
        return success
        
    except Exception as e:
        logger.error(f"Tree-100 test failed: {e}")
        return False

def run_comprehensive_test(args):
    """Run both accuracy test and visualization."""
    logger.info("Running Comprehensive Test Suite...")
    
    success = True
      # Run accuracy test first (unless extreme test is specified)
    if not (hasattr(args, 'extreme') and args.extreme):
        if not run_search_accuracy_test(args):
            success = False
    
    # Run extreme performance test if requested (replaces accuracy test)
    if hasattr(args, 'extreme') and args.extreme:
        if not run_extreme_performance_test(args):
            success = False
    
    # Then launch viewer
    if not launch_database_viewer(args):
        success = False
    
    if success:
        logger.info("Comprehensive test completed successfully!")
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST COMPLETED")
        print("="*60)
        print("✅ Search accuracy test: Complete")
        if hasattr(args, 'extreme') and args.extreme:
            print("✅ Extreme performance test: Complete (110,000+ nodes)")
        else:
            print("✅ Search accuracy test: Complete (1,000+ nodes)")
        print("✅ Database visualization: Available")
        print(f"📁 Results directory: {os.path.abspath('data/')}")
        print("🌐 Streamlit dashboard: http://localhost:8501" if args.streamlit else "📊 Static visualizations: data/visualizations/")
        print("="*60)
    else:
        logger.error("Some components failed - check logs above")
        print("\n❌ Some operations failed - please check the logs")
    
    return success

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="merX Search Accuracy Test and Database Viewer Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,        epilog="""
Examples:
  python launcher.py --test                    # Run search accuracy test only
  python launcher.py --tree100                 # Generate 100-node tree-of-trees structure
  python launcher.py --extreme                 # Run extreme performance test only
  python launcher.py --viewer                  # Launch enhanced 3D database viewer (safe, no errors)
  python launcher.py --enhanced3d              # Launch enhanced 3D viewer with timeline
  python launcher.py --comprehensive           # Run both test and enhanced viewer
  python launcher.py --comprehensive --extreme # Run all tests including extreme performance
  python launcher.py --viewer --streamlit      # Launch interactive Streamlit dashboard
  python launcher.py --install-deps            # Install required dependencies"""
    )
    
    parser.add_argument(
        "--test", action="store_true",
        help="Run search accuracy test with tree-of-trees structure"
    )
    
    parser.add_argument(
        "--tree100", action="store_true",
        help="Generate tree-of-trees test data with exactly 100 nodes"
    )
    
    parser.add_argument(
        "--extreme", action="store_true",
        help="Run extreme performance test with 100,000+ nodes"
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="Launch enhanced 3D database viewer with timeline (safe, no JavaScript errors)"
    )
    
    parser.add_argument(
        "--enhanced3d", action="store_true",
        help="Launch enhanced 3D viewer with timeline and safe node highlighting"
    )
    
    parser.add_argument(
        "--comprehensive", action="store_true",
        help="Run both accuracy test and database viewer"
    )
    
    parser.add_argument(
        "--streamlit", action="store_true",
        help="Use Streamlit for interactive dashboard (with --viewer)"
    )
    
    parser.add_argument(
        "--install-deps", action="store_true",
        help="Install required dependencies"
    )
    
    parser.add_argument(
        "--open-results", action="store_true", default=True,
        help="Automatically open results files/visualizations"
    )
    
    parser.add_argument(
        "--no-open", dest="open_results", action="store_false",
        help="Don't automatically open results"
    )
    
    args = parser.parse_args()    # Show help if no arguments provided
    if not any([args.test, args.tree100, args.extreme, args.viewer, args.enhanced3d, args.comprehensive, args.install_deps]):
        print("merX Memory System - Search Accuracy Test and Database Viewer")
        print("="*60)
        print("🧠 Welcome to the merX comprehensive testing suite!")
        print("")        
        print("Available options:")
        print("  --test            Run search accuracy test (1000+ nodes)")
        print("  --extreme         Run extreme performance test (100,000+ nodes)")
        print("  --viewer          Launch enhanced 3D database viewer (safe, no errors)")
        print("  --enhanced3d      Launch enhanced 3D viewer with timeline")
        print("  --comprehensive   Run everything")
        print("  --streamlit       Use interactive web dashboard")
        print("  --install-deps    Install required packages")
        print("")
        print("Quick start: python launcher.py --comprehensive")
        print("="*60)
        parser.print_help()
        return
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = True
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_requirements():
            success = False    # Run requested components
    if args.comprehensive:
        success = run_comprehensive_test(args)
    else:
        if args.test:
            success = run_search_accuracy_test(args) and success
        
        if args.tree100:
            success = run_tree_100_test(args) and success
        
        if args.extreme:
            success = run_extreme_performance_test(args) and success
        
        if args.viewer or args.enhanced3d:
            success = launch_database_viewer(args) and success
    
    if success:
        logger.info("All requested operations completed successfully!")
    else:
        logger.error("Some operations failed - please check the logs")
        sys.exit(1)

if __name__ == "__main__":
    main()
