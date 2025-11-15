#!/usr/bin/env python3
"""
Generate ArUco markers for robot tracking
Creates both single markers and boards with printable PDFs
"""

import cv2
import numpy as np
import argparse
import yaml
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
    """Generate a ChArUco board (chessboard with ArUco markers)"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    
    # ChArUco board: squaresX and squaresY define the chessboard grid
    # squareLength = marker_size + spacing (total square size)
    # markerLength = marker_size (ArUco marker inside square)
    square_length_px = float(marker_size_px + marker_separation_px)
    marker_length_px = float(marker_size_px)
    
    # Create ChArUco board
    # Note: squaresX/squaresY are the number of squares, not markers
    # For a 3x3 marker grid, we need 4x4 squares (squares = markers + 1)
    charuco_board = cv2.aruco.CharucoBoard(
        (cols, rows),  # squaresX, squaresY (number of squares)
        square_length_px,
        marker_length_px,
        aruco_dict
    )
    
    # Calculate board image size
    # ChArUco: (squaresX * squareLength) x (squaresY * squareLength)
    board_width = int(cols * square_length_px)
    board_height = int(rows * square_length_px)
    
    # Add border (20% of average dimension)
    border_size = int((board_width + board_height) / 2 * 0.2)
    
    # Generate board image
    board_image = charuco_board.generateImage((board_width, board_height), marginSize=border_size)
    
    # Save image
    cv2.imwrite(output_path, board_image)
    print(f"✓ Generated ChArUco board ({rows}x{cols} squares): {output_path}")
    print(f"  Image size: {board_image.shape[1]}x{board_image.shape[0]} pixels")
    print(f"  Square size: {square_length_px:.1f}px, Marker size: {marker_length_px:.1f}px")


def generate_marker_config(output_path: str, marker_type: str, config: dict):
    """Generate YAML configuration file for marker detection"""
    # Minimal config - only what we know for sure + one measurement field
    config_data = {
        'aruco_dict': config.get('aruco_dict', '4x4_50'),
    }
    
    if marker_type == 'single':
        config_data['measured_marker_size'] = None  # User fills this in meters
    elif marker_type == 'board':
        # ChArUco board parameters
        config_data['squares_x'] = config['board_cols']  # squaresX
        config_data['squares_y'] = config['board_rows']  # squaresY
        # Store original values as reference (for calculating scale factor)
        config_data['_original_square_length_meters'] = config['board_marker_size_meters'] + config['board_spacing_meters']
        config_data['_original_marker_length_meters'] = config['board_marker_size_meters']
        # User can measure either marker OR square (square is more accurate but harder to measure)
        config_data['measured_marker_size'] = None  # Option 1: Measure one marker side (easier, assumes uniform scaling)
        config_data['measured_square_size'] = None  # Option 2: Measure one square side (more accurate, optional)
    
    # Write YAML file
    yaml_path = output_path.with_suffix('.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Generated config file: {yaml_path}")
    print(f"  ⚠ After printing, measure ONE marker side and fill in 'measured_marker_size' (in meters)")
    return yaml_path


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
  # Generate default board (4x4, 20cm, 0.6cm spacing)
  python3 generate_marker.py

  # Generate single marker (5cm)
  python3 generate_marker.py --single --id 0 --size 5.0

  # Generate custom board
  python3 generate_marker.py --rows 3 --cols 3 --board-size 16.0 --spacing 0.6

Dictionary types:
  4x4_50    - 50 markers, 4x4 bits (default, good for small markers)
  5x5_100   - 100 markers, 5x5 bits
  6x6_250   - 250 markers, 6x6 bits (better for large markers)
        """
    )
    
    # Marker type (defaults to board if not specified)
    marker_group = parser.add_mutually_exclusive_group(required=False)
    marker_group.add_argument('--single', action='store_true',
                             help='Generate single marker')
    marker_group.add_argument('--board', action='store_true',
                             help='Generate marker board (default)')
    
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
    parser.add_argument('--board-size', type=float, default=20.0,
                       help='Total board physical size in cm (calculates marker size automatically, default: 20.0)')
    parser.add_argument('--marker-size', type=float, default=4.0,
                       help='Individual marker size in cm (default: 4.0, ignored if --board-size is set)')
    parser.add_argument('--spacing', type=float, default=0.6,
                       help='Marker spacing in cm (default: 0.6)')
    
    args = parser.parse_args()
    
    # Default to board if neither --single nor --board specified
    if not args.single and not args.board:
        args.board = True
    
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
        
        # Generate YAML config
        config = {
            'aruco_dict': args.dict,
            'marker_id': args.id,
            'marker_size_meters': args.size / 100.0,  # Convert cm to meters
            'marker_size_cm': args.size,
        }
        generate_marker_config(output_path, 'single', config)
        
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
        
        # Generate YAML config
        config = {
            'aruco_dict': args.dict,
            'board_rows': args.rows,
            'board_cols': args.cols,
            'board_marker_size_meters': marker_size_cm / 100.0,  # Convert cm to meters
            'board_spacing_meters': spacing_cm / 100.0,
            'board_marker_size_cm': marker_size_cm,
            'board_spacing_cm': spacing_cm,
            'total_width_cm': total_width,
            'total_height_cm': total_height,
        }
        generate_marker_config(output_path, 'board', config)
    
    print(f"\n{'='*60}")
    print(f"✓ Generation complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Print the PDF at 100% scale (no 'fit to page')")
    print(f"2. Measure with a ruler:")
    if args.single:
        print(f"   - Measure ONE marker side dimension")
    else:
        print(f"   - Option A (easier): Measure ONE marker side")
        print(f"   - Option B (more accurate): Measure ONE square side (recommended)")
    print(f"3. Edit the YAML config file (*.yaml):")
    if args.single:
        print(f"   - Fill in 'measured_marker_size' (in meters)")
        print(f"   - Example: If marker measures 5.0cm, enter: 0.05")
    else:
        print(f"   - Fill in EITHER 'measured_marker_size' OR 'measured_square_size' (in meters)")
        print(f"   - Example: If marker measures 4.5cm, enter: measured_marker_size: 0.045")
        print(f"   - Example: If square measures 5.1cm, enter: measured_square_size: 0.051")
        print(f"   - All other dimensions will be calculated automatically!")
    print(f"4. Mount marker/board on flat, rigid surface")
    print(f"5. Run marker tracker with: --config <path_to_yaml_file>")
    print()


if __name__ == '__main__':
    main()

