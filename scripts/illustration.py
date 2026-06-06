""" Illustration for the `modeling` package

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: July 2024
Last revised: June 2026

License: GPLv3
"""

import numpy as np
import matplotlib.pyplot as plt

import clustering.tools as tls

#from example import *

#tls.save_ws('data.out',
#    ('COLS', 'DEPTH', 'STEP', 'METHODS',
#     'data', 'index', 'ind_trn', 'ind_gen',
#     'models', 'inp_trn', 'inp_gen'))

tls.load_ws('data.out')


def set_right_yaxis_label(axis, text, **rtext_params):
    """ Write right y-axis side stacked texts """
    esp = "\n     "
    axis_rgt = axis.twinx()
    axis_rgt.tick_params(right=False, labelright=False)
    axis_rgt.yaxis.set_label_position('right')
    axis_rgt.set_ylabel(esp[1:]+esp.join(text), **rtext_params)

def set_margins(axs, **margin_params):
    """ Set the x- and y-axis margins of (a set of) figure axes """
    if np.ndim(axs) == 0:
        axs.margins(**margin_params)
    else:
        for axis in np.ravel(axs):
            axis.margins(**margin_params)

def remove_spaces(fig, no_xspace=False, no_yspace=False):
    """ Remove space between the axes of a figure """
    if no_xspace:
        fig.subplots_adjust(wspace=0.)      # Remove horizontal space
    if no_yspace:
        fig.subplots_adjust(hspace=0.)      # Remove vertical space


##############################################################################
##                           Display the Results                            ##
##############################################################################

# Extract the indexes of the training & the testing data
tstp_trn = index[ind_trn+DEPTH+STEP]     # or `database.index` if `Database`
tstp_gen = index[ind_gen+DEPTH+STEP]     # or `database.index` if `Database`

# Instantiate the figure
fig, axs = plt.subplots(1, len(METHODS),
    figsize=(19.20, 19.20/len(METHODS)), sharey='row')

# Plot the predictions on the training data
for i, model in enumerate(models[0]):
    est_trn = model.predict(inp_trn[0])
    axs[i].plot(tstp_trn, est_trn, '+')

# Plot the predictions on the testing data
for i, model in enumerate(models[0]):
    est_gen = model.predict(inp_gen[0])
    axs[i].plot(tstp_gen, est_gen, '+')

# Axis titles
# fig.suptitle("Multi-modeling with Gating Networks", size=15)
for axis, method in zip(axs, METHODS):
    axis.set_title(method[2], size=15)

# Remove the x & y margins
set_margins(axs, x=0.05, y=0.05)
# Deactivate the axes (and their ticks)
for axis in axs:
    axis.set_xticks([])
    axis.set_yticks([])

# Adjust the plots
plt.tight_layout()
# Remove the spaces between the subplots
remove_spaces(fig, True, True)

# Save the figure in a file
plt.savefig('illustration.pdf', bbox_inches='tight', dpi=300)
plt.close()

##############################################################################
