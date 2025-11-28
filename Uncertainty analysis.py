# Creating a synthetic 3x3 panel figure that mimics the requested UQ figure for fault uncertainty modeling.
# Each of the nine panels is generated separately (to comply with the plotting rules), then combined into
# a single 3x3 image. No seaborn or custom matplotlib styles are used. The code produces and saves the final image
# to /mnt/data/uq_fault_figure.png and individual panels in /mnt/data/
#
# Panels mapping (i)-(i):
# (i.a) prior: 2D interface representation (expected interfaces)
# (i.b) prior: occurrence probability of layer 2
# (i.c) prior: information entropy
# (ii.d) ideal topology (reference model)
# (ii.e) synthetic density model (used as "reference" for gravity forward)
# (ii.f) forward gravity anomaly (synthetic)
# (g) posterior: 2D interface representation (expected interfaces after update)
# (h) posterior: occurrence probability of layer 2
# (i) posterior: information entropy
#
# The figures are synthetic and intended for illustration/visual purposes only.
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ==================== 字体配置(放在最前面) ====================
plt.rcParams.update({
    'font.sans-serif': ['Arial'],  # 黑体
    'axes.unicode_minus': False,  # 解决负号显示
    'font.size': 12,  # 全局字体大小
    'font.weight': 'bold',  # 全局字体加粗
    'axes.labelsize': 9,  # 坐标轴标签字体
    'axes.titlesize': 12,  # 标题字体
    'xtick.labelsize': 12,  # X轴刻度字体
    'ytick.labelsize': 12,  # Y轴刻度字体
    'axes.labelweight': 'bold',  # 坐标轴标签加粗
    'axes.titleweight': 'bold',  # 标题加粗
    'lines.linewidth': 2.5,  # 线条加粗
})
# ============================================================

np.random.seed(42)

# grid settings (cross-section at Y=10000 m)
nx = 300
nz = 200
x = np.linspace(0, 20000, nx)  # meters
z = np.linspace(0, 2000, nz)  # depth meters (positive downward)
X, Z = np.meshgrid(x, z, indexing='xy')


# Reference (true) interfaces - layered model with a normal fault offset
def create_reference_interfaces(offset_x=10000, throw=300):
    # three layers (0: top, 1: layer2, 2: layer3, below)
    base1 = 300 + 0.0005 * (X - 10000) ** 2 * 0  # flat-ish baseline for interface1
    base2 = 800 + 0.00002 * (X - 10000) ** 2
    # introduce a normal-fault-like offset centered at offset_x
    fault_zone = np.exp(-((X - offset_x) ** 2) / (2 * (800 ** 2)))
    # throw applied to the right side of the fault
    throw_profile = throw * fault_zone * (X > offset_x)
    d1 = base1 + throw_profile
    d2 = base2 + throw_profile
    return d1, d2


ref_d1, ref_d2 = create_reference_interfaces()


# Reference layer assignment (for each x, depth position of interfaces)
def build_layer_map(d1, d2):
    layer_map = np.ones((nz, nx), dtype=int) * 3  # default deepest
    for i_x in range(nx):
        z1 = d1[:, i_x] if d1.ndim == 2 else d1[0, i_x]  # but d1 is mesh; we just take column's value function
    # simpler: take mean interface depth per x
    mean_d1 = np.mean(d1, axis=0)
    mean_d2 = np.mean(d2, axis=0)
    for ix in range(nx):
        z1 = mean_d1[ix]
        z2 = mean_d2[ix]
        layer_map[:, ix] = np.where(Z[:, ix] < z1, 0,
                                    np.where(Z[:, ix] < z2, 1, 2))
    return layer_map


ref_layer_map = build_layer_map(ref_d1, ref_d2)

# Generate prior ensemble by perturbing interface depths (random vertical shifts, sigma=300 m)
n_samples = 300
sigma_prior = 300.0  # m
ensemble_layer_maps = np.zeros((n_samples, nz, nx), dtype=np.int8)
for s in range(n_samples):
    shift1 = np.random.normal(0, sigma_prior, size=nx)
    shift2 = np.random.normal(0, sigma_prior, size=nx)
    mean_d1 = np.mean(ref_d1, axis=0) + shift1
    mean_d2 = np.mean(ref_d2, axis=0) + shift2
    # ensure d2 > d1
    mean_d2 = np.maximum(mean_d2, mean_d1 + 50)
    lm = np.zeros((nz, nx), dtype=np.int8)
    for ix in range(nx):
        z1 = mean_d1[ix]
        z2 = mean_d2[ix]
        lm[:, ix] = np.where(Z[:, ix] < z1, 0,
                             np.where(Z[:, ix] < z2, 1, 2))
    ensemble_layer_maps[s] = lm

# Prior probabilities per cell for each layer
prior_probs = np.zeros((3, nz, nx), dtype=float)
for k in range(3):
    prior_probs[k] = np.mean(ensemble_layer_maps == k, axis=0)


# Prior expected interfaces (mean depth where layer transitions occur)
# compute expected depth of interface between layer0-1 and 1-2 from ensemble: median depth where layer changes per x
def expected_interface_from_ensemble(ensemble_maps):
    mean_d1 = np.zeros(nx)
    mean_d2 = np.zeros(nx)
    for ix in range(nx):
        # for each sample find depth index of first transition from 0->1 and 1->2
        d1_list = []
        d2_list = []
        for s in range(n_samples):
            col = ensemble_maps[s, :, ix]
            # find last occurrence of 0 (top) -> depth index
            idx1 = np.max(np.where(col == 0)[0]) if np.any(col == 0) else 0
            idx2 = np.max(np.where(col <= 1)[0]) if np.any(col <= 1) else 0
            d1_list.append(z[idx1])
            d2_list.append(z[idx2])
        mean_d1[ix] = np.median(d1_list)
        mean_d2[ix] = np.median(d2_list)
    return mean_d1, mean_d2


prior_mean_d1, prior_mean_d2 = expected_interface_from_ensemble(ensemble_layer_maps)

# Information entropy for prior
eps = 1e-12
prior_entropy = -np.sum(prior_probs * np.log(prior_probs + eps), axis=0)

# Likelihood representation: (d) ideal topology (reference model map), (e) synthetic density model, (f) forward gravity
# Build synthetic density model from reference layers
densities = np.array([0.0, 200.0, 400.0])  # arbitrary density contrast units
ref_density = np.zeros_like(ref_layer_map, dtype=float)
for k in range(3):
    ref_density[ref_layer_map == k] = densities[k]

# create a synthetic gravity forward: treat each cell as a point mass contribution inversely proportional to depth^2
# Gravity anomaly at surface along x: sum over depths of density / (z + z0)^2
z0 = 50.0
gravity = np.sum(ref_density / (Z + z0) ** 1.5, axis=0)  # a plausible-looking anomaly
# normalize gravity for visualization and then make a 2D XY-style slice by repeating
gravity_2d = np.tile((gravity - gravity.mean()) / gravity.std(), (nz, 1))

# Likelihood map (simple): cells matching reference layer get higher likelihood; mismatch lower.
likelihood = np.where(ref_layer_map == 1, 0.9, 0.1)  # favor layer 1 as example; more nuanced methods possible
# We'll convert likelihood to per-layer likelihoods: if ref layer == k, then p_k = 0.9 else 0.05
likelihood_per_layer = np.zeros_like(prior_probs)
for k in range(3):
    likelihood_per_layer[k] = np.where(ref_layer_map == k, 0.9, 0.05)

# Posterior (unnormalized)
posterior_unnorm = prior_probs * likelihood_per_layer
posterior_probs = posterior_unnorm / (np.sum(posterior_unnorm, axis=0, keepdims=True) + eps)
posterior_entropy = -np.sum(posterior_probs * np.log(posterior_probs + eps), axis=0)

# Posterior expected interfaces: compute depth where posterior most likely changes (pick layer argmax per cell and compute transitions)
post_mode = np.argmax(posterior_probs, axis=0)


def mode_interfaces_from_mode_map(mode_map):
    mean_d1 = np.zeros(nx)
    mean_d2 = np.zeros(nx)
    for ix in range(nx):
        col = mode_map[:, ix]
        idx1 = np.max(np.where(col == 0)[0]) if np.any(col == 0) else 0
        idx2 = np.max(np.where(col <= 1)[0]) if np.any(col <= 1) else 0
        mean_d1[ix] = z[idx1]
        mean_d2[ix] = z[idx2]
    return mean_d1, mean_d2


post_mean_d1, post_mean_d2 = mode_interfaces_from_mode_map(post_mode)


# Utility: a function to plot and save each panel as separate image
def save_panel_image(data, filename, title=None, cmap=None, vmin=None, vmax=None, line_overlays=None):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.imshow(data, origin='lower', aspect='auto', extent=[x.min(), x.max(), z.min(), z.max()], vmin=vmin, vmax=vmax,
               cmap=cmap)
    if title:
        plt.title(title, fontsize=22, fontweight='bold', pad=15)
    plt.xlabel('X (m)', fontsize=18, fontweight='bold')
    plt.ylabel('depth (m)', fontsize=18, fontweight='bold')
    plt.gca().invert_yaxis()

    # 加粗刻度
    plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontweight('bold')

    if line_overlays:
        for line in line_overlays:
            plt.plot(x, line, linewidth=3.0, color='red')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


os.makedirs('./mnt/data/uq_panels', exist_ok=True)

# (a) prior 2D interface representation: plot prior expected interfaces over a faint probability background (layer 0-1 median)
# background: show prior most-likely layer as an image (mode)
prior_mode = np.argmax(prior_probs, axis=0)
save_panel_image(prior_mode, './mnt/data/uq_panels/prior_interface.png', title='(a) Prior: 2D interface (mode)',
                 vmin=0, vmax=2)

# overlay expected interface lines
plt.figure(figsize=(8, 6), dpi=300)
plt.imshow(prior_mode, origin='lower', aspect='auto', extent=[x.min(), x.max(), z.min(), z.max()], vmin=0, vmax=2)
plt.plot(x, prior_mean_d1, linewidth=3.0, color='blue', label='Interface 1')
plt.plot(x, prior_mean_d2, linewidth=3.0, color='green', label='Interface 2')
plt.title('(a) Prior: Two-dimensional interface representation', fontsize=22, fontweight='bold', pad=15)
plt.xlabel('X (m)', fontsize=18, fontweight='bold')
plt.ylabel('depth (m)', fontsize=18, fontweight='bold')
plt.gca().invert_yaxis()
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')
plt.legend(fontsize=14, frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('./mnt/data/uq_panels/prior_interface_overlay.png', dpi=300, bbox_inches='tight')
plt.close()

# (b) prior: probability of layer 2 (index 1)
plt.figure(figsize=(8, 6), dpi=300)
im = plt.imshow(prior_probs[1], origin='lower', aspect='auto', extent=[x.min(), x.max(), z.min(), z.max()],
                vmin=0, vmax=1, cmap='viridis')
plt.title('(b) Prior: Level 2 probability', fontsize=22, fontweight='bold', pad=15)
plt.xlabel('X (m)', fontsize=18, fontweight='bold')
plt.ylabel('depth (m)', fontsize=18, fontweight='bold')
plt.gca().invert_yaxis()
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=14, width=2)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
plt.tight_layout()
plt.savefig('./mnt/data/uq_panels/prior_layer2_prob.png', dpi=300, bbox_inches='tight')
plt.close()

# (c) prior: entropy
plt.figure(figsize=(8, 6), dpi=300)
im = plt.imshow(prior_entropy, origin='lower', aspect='auto', extent=[x.min(), x.max(), z.min(), z.max()],
                vmin=0, vmax=np.log(3), cmap='hot')
plt.title('(c) Prior: Information Entropy', fontsize=22, fontweight='bold', pad=15)
plt.xlabel('X (m)', fontsize=18, fontweight='bold')
plt.ylabel('depth (m)', fontsize=18, fontweight='bold')
plt.gca().invert_yaxis()
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=14, width=2)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
plt.tight_layout()
plt.savefig('./mnt/data/uq_panels/prior_entropy.png', dpi=300, bbox_inches='tight')
plt.close()

# (d) ideal topology (reference map)
save_panel_image(ref_layer_map, './mnt/data/uq_panels/ideal_topology.png',
                 title='(d) Ideal topology (reference model)', vmin=0, vmax=2)

# (e) synthetic density model (reference)
save_panel_image(ref_density, './mnt/data/uq_panels/ref_density.png',
                 title='(e) Synthetic density model (reference)',
                 vmin=densities.min(), vmax=densities.max())

# (f) forward gravity anomaly (XY cross-section like)
save_panel_image(gravity_2d, './mnt/data/uq_panels/forward_gravity.png',
                 title='(f) Forward-modeled gravity anomaly (XY slice)')

# (g) posterior 2D interface representation (mode + overlays)
post_mode_img = np.argmax(posterior_probs, axis=0)
plt.figure(figsize=(8, 6), dpi=300)
plt.imshow(post_mode_img, origin='lower', aspect='auto', extent=[x.min(), x.max(), z.min(), z.max()], vmin=0, vmax=2)
plt.plot(x, post_mean_d1, linewidth=3.0, color='blue', label='Interface 1')
plt.plot(x, post_mean_d2, linewidth=3.0, color='green', label='Interface 2')
plt.title('(g) Posterior: Two-dimensional interface representation', fontsize=22, fontweight='bold', pad=15)
plt.xlabel('X (m)', fontsize=18, fontweight='bold')
plt.ylabel('depth (m)', fontsize=18, fontweight='bold')
plt.gca().invert_yaxis()
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')
plt.legend(fontsize=14, frameon=True, shadow=True)
plt.tight_layout()
plt.savefig('./mnt/data/uq_panels/posterior_interface_overlay.png', dpi=300, bbox_inches='tight')
plt.close()

# (h) posterior: probability of layer 2 (index 1)
plt.figure(figsize=(8, 6), dpi=300)
im = plt.imshow(posterior_probs[1], origin='lower', aspect='auto', extent=[x.min(), x.max(), z.min(), z.max()],
                vmin=0, vmax=1, cmap='viridis')
plt.title('(h) Posterior: Level 2 probability', fontsize=22, fontweight='bold', pad=15)
plt.xlabel('X (m)', fontsize=18, fontweight='bold')
plt.ylabel('depth (m)', fontsize=18, fontweight='bold')
plt.gca().invert_yaxis()
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=14, width=2)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
plt.tight_layout()
plt.savefig('./mnt/data/uq_panels/posterior_layer2_prob.png', dpi=300, bbox_inches='tight')
plt.close()

# (i) posterior: entropy
plt.figure(figsize=(8, 6), dpi=300)
im = plt.imshow(posterior_entropy, origin='lower', aspect='auto', extent=[x.min(), x.max(), z.min(), z.max()],
                vmin=0, vmax=np.log(3), cmap='hot')
plt.title('(i) Posterior: Information Entropy', fontsize=22, fontweight='bold', pad=15)
plt.xlabel('X (m)', fontsize=18, fontweight='bold')
plt.ylabel('depth (m)', fontsize=18, fontweight='bold')
plt.gca().invert_yaxis()
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    label.set_fontweight('bold')
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=14, width=2)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
plt.tight_layout()
plt.savefig('./mnt/data/uq_panels/posterior_entropy.png', dpi=300, bbox_inches='tight')
plt.close()

# Combine into a single 3x3 grid image
panel_files = [
    './mnt/data/uq_panels/prior_interface_overlay.png',
    './mnt/data/uq_panels/prior_layer2_prob.png',
    './mnt/data/uq_panels/prior_entropy.png',
    './mnt/data/uq_panels/ideal_topology.png',
    './mnt/data/uq_panels/ref_density.png',
    './mnt/data/uq_panels/forward_gravity.png',
    './mnt/data/uq_panels/posterior_interface_overlay.png',
    './mnt/data/uq_panels/posterior_layer2_prob.png',
    './mnt/data/uq_panels/posterior_entropy.png',
]

imgs = [Image.open(p) for p in panel_files]
cols = 3;
rows = 3
w, h = imgs[0].size
composite = Image.new('RGB', (cols * w, rows * h), color=(255, 255, 255))
for idx, im in enumerate(imgs):
    r = idx // cols
    c = idx % cols
    composite.paste(im, (c * w, r * h))
out_path = './mnt/data/uq_fault_figure.pdf'
composite.save(out_path, dpi=(300, 300))
print(f"✅ 图像已保存到: {out_path}")


