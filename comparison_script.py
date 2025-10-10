import os
import difflib

def get_file_list(root_dir):
    file_paths = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, root_dir)
            file_paths.add(relative_path)
    return file_paths

def compare_directories(dir1, dir2):
    files1 = get_file_list(dir1)
    files2 = get_file_list(dir2)

    unique_to_dir1 = files1 - files2
    unique_to_dir2 = files2 - files1
    common_files = files1.intersection(files2)

    return sorted(list(unique_to_dir1)), sorted(list(unique_to_dir2)), sorted(list(common_files))

def compare_file_contents(file_path, dir1, dir2):
    path1 = os.path.join(dir1, file_path)
    path2 = os.path.join(dir2, file_path)

    try:
        with open(path1, 'r', encoding='utf-8', errors='ignore') as f1, \
             open(path2, 'r', encoding='utf-8', errors='ignore') as f2:
            content1 = f1.readlines()
            content2 = f2.readlines()
    except FileNotFoundError:
        return None

    diff = list(difflib.unified_diff(content1, content2, fromfile=f'a/{file_path}', tofile=f'b/{file_path}'))
    return diff if diff else None

def main():
    cosmic_crisp_dir = '/Users/benjaminstout/Desktop/CosmicCrisp'
    agent_zero_dir = '/Users/benjaminstout/Desktop/agent-zero-0.9.6'
    report_path = 'comparison_report.md'

    unique_to_cosmic, unique_to_agent_zero, common_files = compare_directories(cosmic_crisp_dir, agent_zero_dir)

    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write("# Project Comparison Report: CosmicCrisp vs. Agent Zero 0.9.6\n\n")

        report_file.write("## Structural Differences\n\n")
        report_file.write("### Files Unique to CosmicCrisp\n\n")
        for file in unique_to_cosmic:
            report_file.write(f"- `{file}`\n")

        report_file.write("\n### Files Unique to Agent Zero 0.9.6\n\n")
        for file in unique_to_agent_zero:
            report_file.write(f"- `{file}`\n")

        report_file.write("\n## Content Differences in Common Files\n\n")
        for file in common_files:
            diff = compare_file_contents(file, cosmic_crisp_dir, agent_zero_dir)
            if diff:
                report_file.write(f"### Differences in `{file}`\n\n")
                report_file.write("```diff\n")
                report_file.writelines(diff)
                report_file.write("\n```\n\n")

    print(f"Comparison report generated at: {report_path}")

if __name__ == "__main__":
    main()
