import argparse

from collections import Counter
from itertools import permutations
from typing import Dict

import geopandas as gpd
import numpy as np
from pandas import concat
from PIL import Image
from PIL.ImageFilter import SMOOTH_MORE
from rasterio.features import shapes
from scipy.spatial.distance import euclidean
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from tqdm import tqdm

import matplotlib.pyplot as plt

# AVAILABLE_COLORS = {
#      1: [228, 23, 23],  # Red-ish,
#      2: [235, 96, 96],  # Lighter Red
#      3: [230, 125, 21], # orange
#      3: [161, 92, 24], # dark orange
#      3: [100, 100, 100], # Yellow
#      3: [254,247,166], # Light Yellow
#      4: [203, 108, 0],  # light brown
#      5: [95, 136, 232], # light blue
#      5: [202, 230, 242], # light blue
#      6: [9, 56, 166],  # dark blue
#      7: [238, 127, 230],  # pink
#      8: [0, 128, 0],  # green
#      8: [5, 77, 9],  # dark green
#      8: [116,213,105],  # light green
#      9: [255, 255, 255],  # white
#      10: [0, 0, 0],  # black
# }

# IDEAS
# Color combinations
#   To understand color combos, take the weighted average of all color combos
#   This quickly adds up to WAY too many colors, even just looking at pairs
#   Is there a quick way we can estimate input a "target color" and determine how much of each
#     other color is needed to arrive at some similar amount of the other color?
# Transform color pallets
#   This isn't as easy as just changing available; Need to transform all pixels equally in some capacity


def open_image(path: str) -> Image:
    image = Image.open(path)
    return image


# image.show()
# image.info


def get_available_colors(image: Image, n_colors=15):
    pixels = list(image.getdata())

    # Cluster (Kmeans) pixels to identify "mean" color of each cluster
    km = KMeans(n_clusters=n_colors, verbose=1, random_state=100)

    km_fit = km.fit(pixels)

    colors = km_fit.cluster_centers_

    color_index_mapping = {}
    for i, color in enumerate(colors):
        color_index_mapping[i] = list(np.round(color).astype(int))

    return color_index_mapping


def map_real_colors_to_available_colors(image: Image, available_colors: Dict) -> Dict:
    # 0: R, 1: G, 2: B
    pixels = list(image.getdata())
    unique_pixel_colors = set(pixels)

    # For each pixel, determine the most similar available colors
    pixel_to_color_dict = {}
    for pixel_iter in tqdm(unique_pixel_colors):
        distances = []
        for avail_iter in available_colors.values():
            distances.append(
                euclidean(
                    pixel_iter,
                    avail_iter,
                )
            )
        min_idx = np.where(distances == np.amin(distances))[0][0]
        pixel_to_color_dict[pixel_iter] = list(available_colors.values())[min_idx]

    return pixel_to_color_dict


def convert_image_to_available_colors(image: Image, color_mapping: Dict) -> Image:
    # Replace all actual pixel values with the closest available color
    image_adj = image.copy()
    width = image_adj.size[0]
    height = image_adj.size[1]
    for i in range(0, width):  # process all pixels
        for j in range(0, height):
            data = image_adj.getpixel((i, j))
            image_adj.putpixel((i, j), tuple(color_mapping[data]))

    return image_adj


def get_neighbors_idx(i, j):

    list(permutations([-1, 0, 1], 2))

    idx_adj = list(permutations([-1, 0, 1], 2))
    idx_adj.extend([[1, 1], [-1, -1]])
    neighbor_idx = []

    for i_adj, j_adj in idx_adj:
        if ((i + i_adj) < 0) or ((j + j_adj) < 0):
            continue
        else:
            neighbor_idx.append((i + i_adj, j + j_adj))

    return neighbor_idx


def get_neighbor_values(image_array: np.array, i, j):

    neighbor_idx = get_neighbors_idx(i, j)
    values = {}
    for i_neb, j_neb in neighbor_idx:
        try:
            values[(i_neb, j_neb)] = image_array[i_neb, j_neb]
        except IndexError:
            continue
    return values


def remove_single_pixels(image: Image) -> Image:
    image_adj = image.copy()
    image_adj_array = np.asarray(image_adj)
    width = image_adj_array.shape[0]
    height = image_adj_array.shape[1]
    for i in tqdm(range(0, width)):  # process all pixels
        for j in range(0, height):
            current_pixel_value = image_adj_array[i][j]

            neb_dict = get_neighbor_values(image_adj_array, i, j)
            # get value of all pixels in 9X9 box
            # neb_dict.update(get_neighbor_values(image_adj_array, i + 1, j))
            # neb_dict.update(get_neighbor_values(image_adj_array, i, j + 1))
            # neb_dict.update(get_neighbor_values(image_adj_array, i - 1, j))
            # neb_dict.update(get_neighbor_values(image_adj_array, i, j - 1))
            neb_values = [tuple(i) for i in list(neb_dict.values())]

            if tuple(current_pixel_value) not in neb_values:
                print(f"Replacing pixel {i} {j}")
                impute_pixel = Counter(neb_values).most_common()[0][0]

                # Note that (j, i) is intentional as opposed to i, j as pillow as a different ordering convetion than numpy
                image_adj.putpixel((j, i), impute_pixel)

    return image_adj


def smooth_small_polygons(image: Image, min_pixels: int = 10):
    # For all pixels, remove groups of pixels that are not of some minimum number of pixels
    # Impute these groups with most prevelant local color

    # This function needs to ...
    #   (1) Identify groups of pixels that are in groups less than min_pixels
    #   (2) Imputes these pixels with the most prevelant local color
    #           - Need some filter to identify the most preveland local color
    return image


def convert_image_to_shapes(image: Image, available_colors: Dict) -> gpd.GeoDataFrame:

    # Image to array
    image_adj_array = np.asarray(image)

    # Create a mask for each available color in the image, convert each to multipolygons
    keep_shapes = []
    for color_id_for_mask in tqdm(available_colors.items()):
        color_for_mask = available_colors[color_id_for_mask[0]]
        image_raster_adj_mask = np.all(image_adj_array == color_for_mask, axis=2)
        all_shapes_color = shapes(image_raster_adj_mask.astype("int32"))

        for shape_iter in all_shapes_color:
            # If shape has value 1 (True), keep it else drop
            if shape_iter[1] == 1:
                geo = gpd.GeoDataFrame(
                    geometry=[Polygon(s) for s in shape_iter[0]["coordinates"]]
                )
                geo["color_index"] = color_id_for_mask[0]
                geo["color_name"] = str(color_for_mask)

                # Set value to color
                # shape_iter[1] = "Color_Placeholder"
                # keep_shapes.append(shape_iter[0])
                keep_shapes.append(geo)

    all_geo = concat(keep_shapes)
    return all_geo


def clean_shapes(gdf, percentile_threshhold=0.05, drop_duplicates=True):

    # Drop shapes with area less than nth percentile; only removes pixel-level noise
    nth_percentile = gdf.geometry.area.quantile(percentile_threshhold)
    gdf = gdf[gdf.geometry.area > nth_percentile]

    # NOTE: buffer(0) is intentional, as this corrects invalid polygons
    gdf["geometry"] = gdf["geometry"].buffer(0)

    if drop_duplicates:
        # Hash geometry as WKB for O(n) duplicate detection instead of O(n²) pairwise comparison
        gdf["_wkb"] = gdf.geometry.apply(lambda g: g.wkb)
        gdf = gdf.drop_duplicates(subset="_wkb", keep="first")
        gdf = gdf.drop(columns="_wkb")

    # Merge same-color shapes into one geometry (handles nesting and adjacency)
    # This eliminates redundant inner/outer polygons of the same color
    gdf = gdf.dissolve(by="color_index", as_index=False)

    # Dissolve can produce MultiPolygons when the same color appears in disconnected
    # regions. Explode them back into individual parts so each part can be labeled.
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # Drop only genuine 1-pixel fragments after dissolve+explode
    nth_percentile = gdf.geometry.area.quantile(0.02)
    gdf = gdf[gdf.geometry.area > nth_percentile].reset_index(drop=True)

    return gdf


def merge_thin_into_neighbors(
    gdf: gpd.GeoDataFrame,
    min_width: float = 20,
    area_retention: float = 0.15,
    max_iter: int = 8,
) -> gpd.GeoDataFrame:
    """Absorb shapes that are too thin to paint into their largest-border neighbor.

    Rather than dropping thin shapes (which leaves blank gaps), this function
    reassigns each thin shape's area to the adjacent thick shape it shares the
    most border with.  The iteration continues until no thin shapes remain or
    max_iter is reached — this naturally handles clusters of thin shapes that
    only border each other: as each is absorbed, the thick shapes grow until
    they reach the remaining isolated thin ones.

    Thin criteria (same as the old remove_thin_shapes):
      1. Erodes to empty — no cross-section wider than min_width.
      2. Area retention < area_retention after inward erosion of min_width/2.
    """
    for iteration in range(max_iter):
        erosion = min_width / 2
        eroded = gdf.geometry.buffer(-erosion)
        original_area = gdf.geometry.area

        is_empty = eroded.is_empty
        retention = eroded.area / original_area.clip(lower=1e-6)
        is_thin = is_empty | (retention < area_retention)

        n_thin = is_thin.sum()
        if n_thin == 0:
            break

        print(f"Iter {iteration + 1}: absorbing {n_thin} thin shapes into neighbors")

        thin_gdf = gdf[is_thin].reset_index(drop=True)
        thick_gdf = gdf[~is_thin].reset_index(drop=True)

        if len(thick_gdf) == 0:
            # All shapes are thin — nothing to absorb into, stop
            break

        # Work on mutable geometry list for thick shapes
        thick_geoms = list(thick_gdf.geometry)

        # Process thin shapes largest-first so big thin shapes get absorbed first,
        # then smaller ones can find newly-enlarged thick neighbors
        thin_order = thin_gdf.geometry.area.sort_values(ascending=False).index

        for thin_idx in thin_order:
            thin_geom = thin_gdf.geometry.iloc[thin_idx]
            # Buffer slightly to reliably detect touching/adjacent shapes
            thin_probe = thin_geom.buffer(1.5)

            best_pos = None
            best_border = 0
            for thick_pos, thick_geom in enumerate(thick_geoms):
                if not thin_probe.intersects(thick_geom):
                    continue
                shared = thin_probe.intersection(thick_geom)
                border_len = shared.length if not shared.is_empty else 0
                if border_len > best_border:
                    best_border = border_len
                    best_pos = thick_pos

            if best_pos is not None:
                thick_geoms[best_pos] = thick_geoms[best_pos].union(thin_geom)

        thick_gdf = thick_gdf.copy()
        thick_gdf["geometry"] = thick_geoms

        # Re-dissolve adjacent same-color shapes that are now touching after absorption
        gdf = (
            thick_gdf.dissolve(by="color_index", as_index=False)
            .explode(index_parts=False)
            .reset_index(drop=True)
        )

    return gdf


def split_large_shapes(
    gdf: gpd.GeoDataFrame, max_area: float = 8000
) -> gpd.GeoDataFrame:
    """Split polygons larger than max_area into grid cells of roughly that size.

    A regular grid is intersected with each oversized polygon.  The cell size is
    chosen so each resulting piece is close to max_area, which keeps both painting
    sections and label placement manageable.  Thin/long shapes naturally become
    a series of shorter, labellable segments.
    """
    from shapely.geometry import box

    cell_size = max_area**0.5  # target cell width ≈ height

    rows = []
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Splitting large shapes"):
        geom = row.geometry
        if geom.area <= max_area:
            rows.append(row)
            continue

        minx, miny, maxx, maxy = geom.bounds
        for x0 in np.arange(minx, maxx, cell_size):
            for y0 in np.arange(miny, maxy, cell_size):
                cell = box(x0, y0, x0 + cell_size, y0 + cell_size)
                piece = geom.intersection(cell)
                if piece.is_empty or piece.area < 1:
                    continue
                new_row = row.copy()
                new_row["geometry"] = piece
                rows.append(new_row)

    return gpd.GeoDataFrame(rows, crs=gdf.crs).reset_index(drop=True)


def smooth_image(image: Image, iter=3) -> Image:
    for _ in range(0, iter):
        image = image.filter(SMOOTH_MORE)
    return image


def render_paint_by_number(
    gdf: gpd.GeoDataFrame, output_path: str = "paint_by_number.png", dpi: int = 200
):
    """Render shapes with color index labels at each polygon's centroid."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Draw all polygons with white fill and thin black outlines
    gdf.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=0.3)

    # Place color_index labels inside each polygon
    # Rules:
    #   1. Every shape MUST have at least one label
    #   2. Large shapes get additional grid labels for readability
    #   3. Grid labels must not fall inside any other (sub-)shape
    #   4. Prefer non-overlapping positions; fall back to best available
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    # Scale label density to image size so labels don't overwhelm large images
    bounds = gdf.geometry.total_bounds  # [minx, miny, maxx, maxy]
    image_extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    label_spacing = max(60, image_extent / 7)  # ~7 labels across the largest dimension
    min_label_dist = label_spacing * 0.4  # no two labels closer than 40% of spacing
    placed_positions = []

    def min_dist_to_placed(x, y):
        if not placed_positions:
            return float("inf")
        return min((x - px) ** 2 + (y - py) ** 2 for px, py in placed_positions) ** 0.5

    def can_place(x, y):
        return min_dist_to_placed(x, y) >= min_label_dist

    def is_inside_other(pt, own_geom, tree, all_geoms):
        """Return True if pt is inside any geometry other than own_geom."""
        for i in tree.query(pt):
            candidate = all_geoms[i]
            if candidate is not own_geom and candidate.contains(pt):
                return True
        return False

    def find_label_pos(geom, own_geom_ref, tree, all_geoms, sample=6):
        """Find the best interior point that avoids other shapes and existing labels.
        Returns (x, y, is_clear) where is_clear=False means overlap was unavoidable."""
        rp = geom.representative_point()

        # Try representative point first
        if can_place(rp.x, rp.y) and not is_inside_other(
            rp, own_geom_ref, tree, all_geoms
        ):
            return rp.x, rp.y, True

        # Sample a grid of interior candidate points
        minx, miny, maxx, maxy = geom.bounds
        step_x = (maxx - minx) / (sample + 1)
        step_y = (maxy - miny) / (sample + 1)
        best_x, best_y = rp.x, rp.y
        best_dist = -1
        for i in range(1, sample + 1):
            for j in range(1, sample + 1):
                x = minx + i * step_x
                y = miny + j * step_y
                pt = Point(x, y)
                if not geom.contains(pt):
                    continue
                if is_inside_other(pt, own_geom_ref, tree, all_geoms):
                    continue
                if can_place(x, y):
                    return x, y, True
                # Track best fallback: point farthest from existing labels
                d = min_dist_to_placed(x, y)
                if d > best_dist:
                    best_dist = d
                    best_x, best_y = x, y

        return best_x, best_y, False

    # Sort by area ascending: small shapes get labeled first
    gdf["_area"] = gdf.geometry.area
    gdf_sorted = gdf.sort_values(by="_area", ascending=True)
    gdf = gdf.drop(columns="_area")

    # Build spatial index of ALL geometries for sub-shape checking
    all_geoms = list(gdf.geometry)
    tree = STRtree(all_geoms)

    for idx, row in gdf_sorted.iterrows():
        geom = row.geometry
        area = geom.area
        label = str(row["color_index"])
        font_size = max(6, min(14, area**0.35))

        if area > label_spacing * label_spacing:
            # Large polygon: place grid labels avoiding sub-shapes
            minx, miny, maxx, maxy = geom.bounds
            x_pts = np.arange(minx + label_spacing / 2, maxx, label_spacing)
            y_pts = np.arange(miny + label_spacing / 2, maxy, label_spacing)
            placed = False
            for x in x_pts:
                for y in y_pts:
                    pt = Point(x, y)
                    if not geom.contains(pt) or not can_place(x, y):
                        continue
                    if not is_inside_other(pt, geom, tree, all_geoms):
                        ax.text(
                            x,
                            y,
                            label,
                            ha="center",
                            va="center",
                            fontsize=font_size,
                            color="black",
                        )
                        placed_positions.append((x, y))
                        placed = True
            # Guarantee at least one label per shape
            if not placed:
                x, y, _ = find_label_pos(geom, geom, tree, all_geoms)
                ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="black",
                )
                placed_positions.append((x, y))
        else:
            # Small/medium polygon: find best non-overlapping interior point
            x, y, _ = find_label_pos(geom, geom, tree, all_geoms)
            ax.text(
                x, y, label, ha="center", va="center", fontsize=font_size, color="black"
            )
            placed_positions.append((x, y))

    ax.set_aspect("equal")
    ax.axis("off")

    # Flip y-axis since image coordinates have origin at top-left
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Paint-by-number saved to {output_path}")


def render_colored_result(
    gdf: gpd.GeoDataFrame, output_path: str = "image_colored.png", dpi: int = 200
):
    """Render the cleaned shapes filled with their actual colors.

    Lets you visually verify what the final painted result will look like
    after all shape cleaning and thin-shape merging has been applied.
    """
    import re

    def parse_color(color_name: str):
        # Handles numpy repr "np.int64(255)" and plain Python repr "255"
        nums = re.findall(
            r"\((\d+)\)", color_name
        )  # numpy format: grab value inside ()
        if not nums:
            nums = re.findall(r"\d+", color_name)  # plain int format
        return tuple(int(x) / 255.0 for x in nums[:3])

    _, ax = plt.subplots(1, 1, figsize=(16, 12))

    colors = [parse_color(row["color_name"]) for _, row in gdf.iterrows()]
    gdf.plot(ax=ax, color=colors, edgecolor="none")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Colored result saved to {output_path}")


def main():

    # create parser
    parser = argparse.ArgumentParser()

    # add arguments to the parser

    parser.add_argument("--input_path", help="Path to input file (jpg, png)")
    parser.add_argument(
        "--layer_name", default="picture_layer", help="Name of layer in qgis layer"
    )
    parser.add_argument(
        "--output_path",
        default="paint_by_number.png",
        help="Path to output paint-by-number image (png, pdf)",
    )

    # parse the arguments
    args = parser.parse_args()

    image = open_image(args.input_path)
    available_colors = get_available_colors(image, 10)
    image = smooth_image(
        image, 8
    )  # Helpful when there are very fine details; Can I programatically identify?
    pixel_to_color_dict = map_real_colors_to_available_colors(image, available_colors)
    image_adj = convert_image_to_available_colors(image, pixel_to_color_dict)

    image_adj_new = remove_single_pixels(image_adj)
    image_adj_new.save("image_adj.png")
    image_adj_gdf = convert_image_to_shapes(image_adj_new, available_colors)
    image_adj_gdf = clean_shapes(image_adj_gdf.copy())
    image_adj_gdf = merge_thin_into_neighbors(image_adj_gdf, min_width=8)

    # Remove shapes too small to physically paint (< ~7x7 pixels)
    min_paintable_area = 50
    n_before = len(image_adj_gdf)
    image_adj_gdf = image_adj_gdf[
        image_adj_gdf.geometry.area >= min_paintable_area
    ].reset_index(drop=True)
    n_removed = n_before - len(image_adj_gdf)
    if n_removed:
        print(
            f"Removed {n_removed} unpaintable shapes (area < {min_paintable_area}px²)"
        )

    # Render colored preview (what the finished painting will look like)
    render_colored_result(image_adj_gdf, "image_colored.png")

    # Render paint-by-number image with labeled shapes
    render_paint_by_number(image_adj_gdf, args.output_path)


if __name__ == "__main__":
    main()
