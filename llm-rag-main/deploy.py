#!/usr/bin/env python3
"""
Deployment script for Enhanced Legal AI System.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed."""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'rouge-score'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} (missing)")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("✅ All packages installed successfully")
    else:
        print("✅ All requirements satisfied")

def run_tests():
    """Run comprehensive tests."""
    print("\n🧪 Running comprehensive tests...")
    
    test_file = Path("tests/test_all_components.py")
    if not test_file.exists():
        print("❌ Test file not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def start_web_app():
    """Start the enhanced web application."""
    print("\n🚀 Starting Enhanced Legal AI Web Application...")
    
    app_file = Path("web_app/enhanced_app.py")
    if not app_file.exists():
        print("❌ Web app file not found")
        return False
    
    try:
        print("🌐 Opening web application in browser...")
        print("📍 URL: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the server")
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            str(app_file), '--server.port', '8501'
        ], cwd=Path.cwd())
        
    except KeyboardInterrupt:
        print("\n🛑 Web application stopped")
    except Exception as e:
        print(f"❌ Error starting web app: {e}")
        return False
    
    return True

def main():
    """Main deployment function."""
    print("🚀 ENHANCED LEGAL AI SYSTEM DEPLOYMENT")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Step 1: Check requirements
    check_requirements()
    
    # Step 2: Run tests
    if not run_tests():
        print("\n❌ Deployment failed: Tests did not pass")
        return False
    
    # Step 3: Start web application
    print("\n🎉 System ready for deployment!")
    
    user_input = input("\n🚀 Start web application? (y/n): ").lower().strip()
    if user_input in ['y', 'yes']:
        start_web_app()
    else:
        print("\n📋 To start manually, run:")
        print("   streamlit run web_app/enhanced_app.py")
    
    print("\n✅ Deployment completed successfully!")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎯 Enhanced Legal AI System is ready!")
        else:
            print("\n❌ Deployment failed. Please check the errors above.")
    except KeyboardInterrupt:
        print("\n🛑 Deployment cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
