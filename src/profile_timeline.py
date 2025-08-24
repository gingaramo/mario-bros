import argparse
import json
import os
import shutil
import glob


def find_profile_files(prefix):
  """
    Find all JSON files that match the given prefix.
    
    Args:
        prefix (str): The prefix to match against filenames.
        
    Returns:
        list: List of matching file paths.
    """
  # Handle both absolute and relative paths
  if os.path.isabs(prefix):
    pattern = f"{prefix}*.json"
  else:
    pattern = f"{prefix}*.json"

  matching_files = glob.glob(pattern)
  return sorted(matching_files)


def create_timeline_html(profile_prefix, output_dir="timeline_output"):
  """
    Reads event data from JSON files matching the prefix and generates an HTML timeline visualization
    with separate HTML, CSS, and JavaScript files.

    Args:
        profile_prefix (str): The prefix to match profile files against.
        output_dir (str): Directory to save the output files.
    """
  # Find all matching profile files
  profile_files = find_profile_files(profile_prefix)

  if not profile_files:
    print(f"Error: No JSON files found matching prefix '{profile_prefix}'")
    print(f"Tried pattern: {profile_prefix}*.json")
    return

  print(
      f"Found {len(profile_files)} profile files matching '{profile_prefix}':")
  for file_path in profile_files:
    print(f"  - {file_path}")

  # Load data from all matching files
  all_data = {}
  for file_path in profile_files:
    try:
      with open(file_path, 'r') as f:
        data = json.load(f)

      # Use the filename (without extension) as the profile name
      profile_name = os.path.splitext(os.path.basename(file_path))[0]
      all_data[profile_name] = data

    except FileNotFoundError as e:
      print(f"Warning: Could not read file {file_path}: {e}")
      continue
    except json.JSONDecodeError as e:
      print(f"Warning: Invalid JSON in file {file_path}: {e}")
      continue

  if not all_data:
    print("Error: No valid profile data could be loaded.")
    return

  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # Get the directory where this script is located
  script_dir = os.path.dirname(os.path.abspath(__file__))

  # Copy the static files to the output directory
  static_files = ['timeline.html', 'timeline.css', 'timeline.js']
  for file_name in static_files:
    src_path = os.path.join(script_dir, file_name)
    dst_path = os.path.join(output_dir, file_name)
    if os.path.exists(src_path):
      shutil.copy2(src_path, dst_path)
    else:
      print(f"Warning: Static file {file_name} not found in {script_dir}")

  # Create a data.js file with the timeline data
  data_js_content = f"""// Timeline data
const timelineData = {json.dumps(all_data, indent=2)};

// Initialize the timeline when the page loads
document.addEventListener('DOMContentLoaded', function() {{
    initializeTimeline(timelineData);
}});
"""

  data_js_path = os.path.join(output_dir, 'data.js')
  with open(data_js_path, 'w') as f:
    f.write(data_js_content)

  # Update the HTML file to include the data.js script
  html_path = os.path.join(output_dir, 'timeline.html')
  if os.path.exists(html_path):
    with open(html_path, 'r') as f:
      html_content = f.read()

    # Add the data.js script before the closing body tag
    updated_html = html_content.replace(
        '<script src="timeline.js"></script>',
        '<script src="timeline.js"></script>\n    <script src="data.js"></script>'
    )

    with open(html_path, 'w') as f:
      f.write(updated_html)

  print(f"Timeline visualization created in directory: {output_dir}")
  print(
      f"Open {os.path.join(output_dir, 'timeline.html')} in your browser to view the timeline."
  )


def create_argument_parser():
  """
    Create and configure the command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
  parser = argparse.ArgumentParser(
      description="Reinforcement Learning Training Profile Timeline Generator",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--profile',
      type=str,
      required=True,
      help=
      'Prefix to match profile JSON files (e.g., "foo_" matches foo_bar.json, foo_baz.json, etc.)'
  )
  parser.add_argument('--output',
                      type=str,
                      default='timeline_output',
                      help='Output directory for the timeline files')
  return parser


if __name__ == "__main__":
  parser = create_argument_parser()
  args = parser.parse_args()

  if not args.profile:
    print("Please provide the --profile argument with a prefix to match.")
    print("Example: python profile_timeline.py --profile foo_")
    print("This will match all files like foo_bar.json, foo_baz.json, etc.")
  else:
    create_timeline_html(args.profile, args.output)
