from pylab import *

params = {
   'axes.labelsize': 15,
   'text.fontsize': 15,
   'legend.fontsize': 15,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
#   'text.usetex': True,
#   'figure.figsize': fig_size
}

rcParams.update(params)



def load_data( fname ):
   f = open( fname )
   data = np.loadtxt( f, delimiter=',' )
   f.close
   x, y = zip( *data )
   return x, y



def plot_kind( k, m, c ):
   x,y = load_data("d:\\data" + k )
   #plot( x, y, m )
   scatter( x, y, marker=m, color=c, label=k )


#plot_kind( "tm_f", 'x' )
if 1:
   plot_kind( "gm_f",  's', 'r' )
   plot_kind( "lf_m",  's', 'g' )
   plot_kind( "mv_f",  's', 'b' )
   plot_kind( "pe_m",  's', 'c' )
   plot_kind( "pf_f",  'o', 'm' )
   plot_kind( "pg_f",  'o', 'y' )
   plot_kind( "tg_f",  'o', 'k' )
   plot_kind( "tg_m",  'o', 'r' )
   plot_kind( "tm_f",  '+', 'g' )
   plot_kind( "tm_m",  '+', 'b' )
   plot_kind( "toc_f", '+', 'c' )
   plot_kind( "toc_m", '+', 'm' )

legend( ncol=1, loc='center right',
   bbox_to_anchor=(1.3,0.5), scatterpoints=1 )

# Axis stuff
#axis([1,7,52,70])
xlabel("Hue (Most frequent)")
ylabel("Hue (Second most frequent)")

# adjust margins
subplots_adjust( left=0.13, right=0.8, bottom=0.1, top=0.95 )

if 1:
   # show on screen
   show()
else:
   # save as eps
   savefig( "h1h2.png" )