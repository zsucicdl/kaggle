# Method 1: Using sys module
import sys
print(sys.version)  # Full version info
print(sys.version_info)  # Version info as a tuple

# Method 2: Using platform module
import platform
print(platform.python_version())  # Just the version number

# Method 3: Using os through command line
import os
print(os.system('python --version'))  # Or 'python3 --version' on some systems