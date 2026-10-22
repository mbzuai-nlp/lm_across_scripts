# read a file and extract lines up to a certain number of words
# and write them to a new file

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        '--tokenized_file',
                        required=True,
                        help='source file')
    parser.add_argument('-i',
                        '--input_file',
                        required=True,
                        help='target file')
    parser.add_argument('-w',
                        '--max_words',
                        type=int,
                        help='max words per line')
    args = parser.parse_args()

    with open(args.tokenized_file, 'r') as f:
        print('reading from', args.tokenized_file)
        source_lines = f.readlines()
    
    with open(args.input_file, 'r') as f:
        print('reading from', args.input_file)
        target_lines = f.readlines()

    # use m for million
    max_words_string = f'{args.max_words // 1000000}m' if args.max_words >= 1000000 else str(args.max_words)

    os.makedirs(f'{max_words_string}', exist_ok=True)
    source_output_file = os.path.join(max_words_string, args.tokenized_file)
    target_output_file = os.path.join(max_words_string, args.input_file)
    
    os.makedirs(f'test', exist_ok=True)
    os.makedirs(f'dev', exist_ok=True)
    
    test_words = 1000000
    # open both files using zip
    
    test_source_output_file = os.path.join('test', args.tokenized_file)
    test_target_output_file = os.path.join('test', args.input_file)

    test_line = 0
    with open(test_source_output_file, 'w') as f1, open(test_target_output_file, 'w') as f2:
        current_words = 0
        for i, (source_line, target_line) in enumerate(zip(source_lines, target_lines)):
            source_words = source_line.split()
            if current_words + len(source_words) < test_words:
                f1.write(source_line)
                f2.write(target_line)
                current_words += len(source_words)
            else:
                test_line = i
                print(f'reached max words per line at {current_words} in the source at line {test_line}')
                break

    dev_words = 1000000
    # open both files using zip
    
    dev_source_output_file = os.path.join('dev', args.tokenized_file)
    dev_target_output_file = os.path.join('dev', args.input_file)

    print(f'Skipping the frist {test_line} lines...')
    dev_line = 0
    with open(dev_source_output_file, 'w') as f1, open(dev_target_output_file, 'w') as f2:
        current_words = 0
        for i, (source_line, target_line) in enumerate(zip(source_lines[test_line:], target_lines[test_line:])):
            source_words = source_line.split()
            if current_words + len(source_words) < dev_words:
                f1.write(source_line)
                f2.write(target_line)
                current_words += len(source_words)
            else:
                dev_line = i + test_line
                print(f'reached max words per line at {current_words} in the source at line {dev_line}')
                break

    print(f'Skipping the frist {dev_line} lines...')
   # open both files using zip
    with open(source_output_file, 'w') as f1, open(target_output_file, 'w') as f2:
        current_words = 0
        for source_line, target_line in zip(source_lines[dev_line:], target_lines[dev_line:]):
            source_words = source_line.split()
            if current_words + len(source_words) < args.max_words:
                f1.write(source_line)
                f2.write(target_line)
                current_words += len(source_words)
            else:
                print(f'reached max words per line at {current_words} in the source')
                break


if __name__ == '__main__':
    main()