import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from pylab import rcParams


rcParams['figure.figsize'] = 12, 10


markers = []
for m in Line2D.markers:
  try:
    if len(m) == 1 and m != ' ':
      markers.append(m)
  except TypeError:
    pass

STYLES = markers + [
  r'$\lambda$',
  r'$\bowtie$',
  r'$\circlearrowleft$',
  r'$\clubsuit$',
  r'$\checkmark$']

STYLES *= 10

# --- DATA ----
solutions_diffs = np.load('diffs.npy')
tolerances = np.linspace(1e-3, 0.5, num=50)

# --- PLOT ----
def plot_diffs_vs_tolerances(tolerances=tolerances,
                             solutions_diffs=solutions_diffs):
    plt.close()
    converged_diffs = [diffs[-1] for diffs in solutions_diffs]
    plt.plot(tolerances, converged_diffs,
             '-o',
             color='b', lw=1.5,
             label='|SGP-GP|')

    plt.ylabel('diffirence bw sketched GP and GP', fontsize=24)
    plt.xlabel('tolerances', fontsize=24)
    plt.xticks(fontsize=18)
    # plt.locator_params(nbins=11, axis='x')
    plt.yticks(fontsize=18)
    # plt.locator_params(nbins=11, axis='y')
    plt.legend(fontsize=17.5,
               # loc=(1.04, 0)
               )
    plt.grid()
    plt.tight_layout()
    # plt.title('GP vs SGP', fontsize=30)
    plt.savefig('diff_vs_tolerances')
    # plt.show()


def plot_diffs_patterns(tolerances=tolerances,
                        solutions_diffs=solutions_diffs):
    plt.close()
    rcParams['figure.figsize'] = 18, 12.5
    COLORS = iter(cm.rainbow(np.linspace(0, 1, 50)))

    for i in range(1, len(solutions_diffs)):
      diffs = solutions_diffs[i]
      color = next(COLORS)
      tolerance = tolerances[i]
      label = '{}'.format(tolerance)
      marker = STYLES[-i]
      plt.plot(range(len(diffs)),
               diffs,
               '-d',
               marker=marker, markersize=4,
               color=color,
               lw=0.8,
               label=label
               )

    plt.ylabel('diffirence bw sketched GP and GP', fontsize=24)
    plt.xlabel('epochs * 10', fontsize=24)
    plt.xticks(fontsize=24)
    plt.locator_params(nbins=11, axis='x')
    plt.yticks(fontsize=24)
    # plt.locator_params(nbins=11, axis='y')
    # plt.legend(fontsize=12.5,
    #            loc=(1.04, 0))
    plt.legend(loc=(1.02, 0), fontsize=9.75)
    plt.grid()
    # plt.title('RESNET83v2', fontsize=30)
    plt.tight_layout()
    plt.savefig('diff_patterns')
    # plt.show()


plot_diffs_vs_tolerances()
# plot_diffs_patterns()
