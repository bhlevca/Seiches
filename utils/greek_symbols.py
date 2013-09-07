for greek_code in range(0x3b1, 0x3ca):
    greek_char = unichr(greek_code).encode('utf-8')
    print hex(greek_code), greek_char
