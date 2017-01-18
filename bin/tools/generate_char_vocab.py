#! /usr/bin/env python

"""
Generate characyer vocabulary for a text file.
"""

import fileinput

def main():
  """Main function"""
  chars = set()

  for line in fileinput.input():
    for char in line.strip():
      chars.add(char)

  print("\n".join(sorted(list(chars))))

if __name__ == "__main__":
  main()
