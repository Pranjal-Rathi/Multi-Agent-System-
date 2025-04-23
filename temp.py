from PIL import Image
from decimal import Decimal, getcontext
import pickle
import sys

# Increase Decimal precision so that the arithmetic encoding works on a moderate amount of data.
getcontext().prec = 150  # Increase as needed for larger images

def calculate_frequencies(data):
    """Calculate the frequency (count) of each symbol in the data and return a probability model."""
    freq = {}
    for symbol in data:
        freq[symbol] = freq.get(symbol, 0) + 1
    total = len(data)
    probabilities = {symbol: Decimal(count) / Decimal(total) for symbol, count in freq.items()}
    return probabilities

def build_cumulative_intervals(probabilities):
    """Build a cumulative probability (CDF) interval mapping for each symbol.
       Returns a dictionary mapping symbol -> (low, high)
    """
    intervals = {}
    cumulative = Decimal(0)
    for symbol, prob in sorted(probabilities.items()):
        low = cumulative
        high = cumulative + prob
        intervals[symbol] = (low, high)
        cumulative = high
    return intervals

def arithmetic_encode(data, intervals):
    """Encode a sequence of symbols using arithmetic coding.
       data: list of symbols.
       intervals: cumulative intervals for each symbol.
       Returns: a Decimal code in the final interval.
    """
    low = Decimal(0)
    high = Decimal(1)
    for symbol in data:
        range_width = high - low
        sym_low, sym_high = intervals[symbol]
        high = low + range_width * sym_high
        low = low + range_width * sym_low
    # Any value in [low, high) is a valid encoding. We use the midpoint.
    return (low + high) / 2

def arithmetic_decode(code, data_length, intervals):
    """Decode an arithmetic-coded value given:
         code: the encoded Decimal value.
         data_length: number of symbols to decode.
         intervals: cumulative intervals (same as used in encoding).
       Returns a list of decoded symbols.
    """
    decoded = []
    low = Decimal(0)
    high = Decimal(1)
    
    # To allow lookup of intervals, sort symbols by their interval ranges (by low value)
    sorted_symbols = sorted(intervals.items(), key=lambda item: item[1][0])
    
    for _ in range(data_length):
        range_width = high - low
        # Determine which symbol’s sub-interval contains the code
        for symbol, (sym_low, sym_high) in sorted_symbols:
            candidate_low = low + range_width * sym_low
            candidate_high = low + range_width * sym_high
            if candidate_low <= code < candidate_high:
                decoded.append(symbol)
                # Narrow the range to the candidate
                low = candidate_low
                high = candidate_high
                break
    return decoded

def compress_image(image_path, compressed_path):
    """Compress an image file using arithmetic coding (grayscale)."""
    # Open image and convert to grayscale for simplicity.
    image = Image.open(image_path).convert("L")
    width, height = image.size
    data = list(image.getdata())
    
    # Build probability model and cumulative intervals.
    probabilities = calculate_frequencies(data)
    intervals = build_cumulative_intervals(probabilities)
    
    # Encode the pixel values using arithmetic coding.
    code = arithmetic_encode(data, intervals)
    
    # Save compressed data (code, original length, probabilities, image dimensions)
    compressed_data = {
        "code": str(code),  # save as string to preserve precision
        "length": len(data),
        "probabilities": {str(k): str(v) for k, v in probabilities.items()},
        "size": (width, height)
    }
    with open(compressed_path, "wb") as f:
        pickle.dump(compressed_data, f)
    print(f"Compressed image saved to {compressed_path}")

def decompress_image(compressed_path, output_image_path):
    """Decompress an image file that was compressed using arithmetic coding."""
    with open(compressed_path, "rb") as f:
        compressed_data = pickle.load(f)
    
    # Retrieve stored data
    code = Decimal(compressed_data["code"])
    length = compressed_data["length"]
    size = tuple(compressed_data["size"])
    # Recover probabilities: keys are saved as strings, convert them to integer symbols.
    probabilities = {int(k): Decimal(v) for k, v in compressed_data["probabilities"].items()}
    intervals = build_cumulative_intervals(probabilities)
    
    decoded_data = arithmetic_decode(code, length, intervals)
    image = Image.new("L", size)
    image.putdata(decoded_data)
    image.save(output_image_path)
    print(f"Decompressed image saved to {output_image_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Image compression and decompression using arithmetic coding.")
    parser.add_argument("mode", choices=["compress", "decompress"], help="Operation mode.")
    parser.add_argument("input", help="Input file path (image file for compress, compressed file for decompress).")
    parser.add_argument("output", help="Output file path (compressed file for compress, image file for decompress).")
    args = parser.parse_args()
    
    if args.mode == "compress":
        compress_image(args.input, args.output)
    else:
        decompress_image(args.input, args.output)

if __name__ == "__main__":
    main()
