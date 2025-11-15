#!/usr/bin/env python3
"""
Generate ArUco markers for robot tracking
Creates both single markers and boards with printable PDFs
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def generate_single_marker(marker_id: int, marker_size: int, dict_type: int, output_path: str):
    """Generate a single ArUco marker"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    
    # Generate marker image
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # Add white border (20% of marker size)
    border_size = int(marker_size * 0.2)
    marker_with_border = cv2.copyMakeBorder(
        marker_image, border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT, value=255
    )
    
    # Save image
    cv2.imwrite(output_path, marker_with_border)
    print(f"✓ Generated single marker (ID={marker_id}): {output_path}")
    print(f"  Image size: {marker_with_border.shape[0]}x{marker_with_border.shape[1]} pixels")


def generate_board(rows: int, cols: int, marker_size_px: int, marker_separation_px: int,
                   dict_type: int, first_marker_id: int, output_path: str):
    """Generate an ArUco board (grid of markers)"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    
    # Create board
    board = cv2.aruco.GridBoard(
        (cols, rows),
        float(marker_size_px),
        float(marker_separation_px),
        aruco_dict
    )
    
    # Calculate board image size
    board_width = cols * marker_size_px + (cols - 1) * marker_separation_px
    board_height = rows * marker_size_px + (rows - 1) * marker_separation_px
    
    # Add border (20% of average dimension)
    border_size = int((board_width + board_height) / 2 * 0.2)
    
    # Generate board image
    board_image = board.generateImage((board_width, board_height), marginSize=border_size)
    
    # Save image
    cv2.imwrite(output_path, board_image)
    print(f"✓ Generated {rows}x{cols} board: {output_path}")
    print(f"  Image size: {board_image.shape[1]}x{board_image.shape[0]} pixels")
    print(f"  Marker IDs: {first_marker_id} to {first_marker_id + rows*cols - 1}")


def create_printable_pdf(image_path: str, physical_size_cm: float, output_pdf: str):
    """Create a printable PDF with exact physical dimensions"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.pdfgen import canvas
        from PIL import Image
        
        # Load image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Create PDF
        c = canvas.Canvas(output_pdf, pagesize=A4)
        page_width, page_height = A4
        
        # Calculate dimensions maintaining aspect ratio
        aspect_ratio = img_height / img_width
        pdf_width = physical_size_cm * cm
        pdf_height = pdf_width * aspect_ratio
        
        # Center on page
        x_offset = (page_width - pdf_width) / 2
        y_offset = (page_height - pdf_height) / 2
        
        # Draw image
        c.drawImage(image_path, x_offset, y_offset, width=pdf_width, height=pdf_height)
        
        # Add info text
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, page_height - 1*cm, f"ArUco Marker - Print at {physical_size_cm}cm width")
        c.drawString(2*cm, page_height - 1.5*cm, f"Actual size: {physical_size_cm:.1f}cm x {physical_size_cm * aspect_ratio:.1f}cm")
        c.drawString(2*cm, 1*cm, "⚠ Print at 100% scale - DO NOT scale to fit page")
        
        c.save()
        print(f"✓ Generated printable PDF: {output_pdf}")
        print(f"  Physical size: {physical_size_cm:.1f}cm x {physical_size_cm * aspect_ratio:.1f}cm")
        print(f"  ⚠ IMPORTANT: Print at 100% scale (no 'fit to page')")
        
    except ImportError:
        print("⚠ reportlab not installed. PDF generation skipped.")
        print("  Install with: pip install reportlab pillow")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate ArUco markers for robot tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single marker (5cm)
  python3 generate_marker.py --single --id 0 --size 5.0

  # Generate 4x4 board (20cm total)
  python3 generate_marker.py --board --rows 4 --cols 4 --board-size 20.0

  # Generate board with custom marker size
  python3 generate_marker.py --board --rows 3 --cols 3 --marker-size 4.0 --spacing 1.0

Dictionary types:
  4x4_50    - 50 markers, 4x4 bits (default, good for small markers)
  5x5_100   - 100 markers, 5x5 bits
  6x6_250   - 250 markers, 6x6 bits (better for large markers)
        """
    )
    
    # Marker type
    marker_group = parser.add_mutually_exclusive_group(required=True)
    marker_group.add_argument('--single', action='store_true',
                             help='Generate single marker')
    marker_group.add_argument('--board', action='store_true',
                             help='Generate marker board')
    
    # Common parameters
    parser.add_argument('--dict', type=str, default='4x4_50',
                       choices=['4x4_50', '5x5_100', '6x6_250'],
                       help='ArUco dictionary (default: 4x4_50)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Print DPI for size calculation (default: 300)')
    parser.add_argument('--output-dir', type=str, default='markers',
                       help='Output directory (default: markers/)')
    
    # Single marker parameters
    parser.add_argument('--id', type=int, default=0,
                       help='Marker ID for single marker (default: 0)')
    parser.add_argument('--size', type=float, default=5.0,
                       help='Physical marker size in cm (default: 5.0)')
    
    # Board parameters
    parser.add_argument('--rows', type=int, default=4,
                       help='Board rows (default: 4)')
    parser.add_argument('--cols', type=int, default=4,
                       help='Board columns (default: 4)')
    parser.add_argument('--board-size', type=float,
                       help='Total board physical size in cm (calculates marker size automatically)')
    parser.add_argument('--marker-size', type=float, default=4.0,
                       help='Individual marker size in cm (default: 4.0)')
    parser.add_argument('--spacing', type=float, default=1.0,
                       help='Marker spacing in cm (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Dictionary mapping
    dict_map = {
        '4x4_50': cv2.aruco.DICT_4X4_50,
        '5x5_100': cv2.aruco.DICT_5X5_100,
        '6x6_250': cv2.aruco.DICT_6X6_250,
    }
    dict_type = dict_map[args.dict]
    
    # Convert cm to pixels at specified DPI
    cm_to_px = args.dpi / 2.54
    
    print(f"\n{'='*60}")
    print(f"ArUco Marker Generator")
    print(f"{'='*60}")
    print(f"Dictionary: {args.dict}")
    print(f"Output directory: {output_dir}")
    print(f"DPI: {args.dpi}")
    print(f"{'='*60}\n")
    
    if args.single:
        # Generate single marker
        marker_size_px = int(args.size * cm_to_px)
        output_path = output_dir / f"marker_id{args.id}_{args.dict}_{args.size:.1f}cm.png"
        
        generate_single_marker(args.id, marker_size_px, dict_type, str(output_path))
        
        # Generate PDF
        pdf_path = output_path.with_suffix('.pdf')
        create_printable_pdf(str(output_path), args.size, str(pdf_path))
        
    else:
        # Generate board
        if args.board_size:
            # Calculate marker size from total board size
            # board_size = rows * marker_size + (rows-1) * spacing
            marker_size_cm = (args.board_size - (max(args.rows, args.cols) - 1) * args.spacing) / max(args.rows, args.cols)
            spacing_cm = args.spacing
        else:
            marker_size_cm = args.marker_size
            spacing_cm = args.spacing
        
        marker_size_px = int(marker_size_cm * cm_to_px)
        spacing_px = int(spacing_cm * cm_to_px)
        
        total_width = args.cols * marker_size_cm + (args.cols - 1) * spacing_cm
        total_height = args.rows * marker_size_cm + (args.rows - 1) * spacing_cm
        
        output_path = output_dir / f"board_{args.rows}x{args.cols}_{args.dict}_{total_width:.1f}x{total_height:.1f}cm.png"
        
        generate_board(
            args.rows, args.cols, marker_size_px, spacing_px,
            dict_type, 0, str(output_path)
        )
        
        # Generate PDF
        pdf_path = output_path.with_suffix('.pdf')
        create_printable_pdf(str(output_path), total_width, str(pdf_path))
    
    print(f"\n{'='*60}")
    print(f"✓ Generation complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Print the PDF at 100% scale (no 'fit to page')")
    print(f"2. Verify physical dimensions with a ruler")
    print(f"3. Mount on flat, rigid surface")
    print(f"4. Run marker tracker with matching --marker-size parameter")
    print()


if __name__ == '__main__':
    main()

