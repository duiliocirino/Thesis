import matplotlib.pyplot as plt
import utility


# legend_names = [r'$\Pi_{\mathcal{B}}$ MHE', r'$\Pi_{\mathcal{B}}$ MHE Reg', r'$\Pi_{\mathcal{B}}$ MSE', r'$\Pi_{\tilde \mathcal{S}}$ MHE', r'$\Pi_{\tilde \mathcal{S}}$ MHE Reg', r'$\Pi_{\tilde \mathcal{S}}$ MSE'
# legend_names = [r'$\rho=0$', r'$\rho=0.0002$', r'$\rho=0.0005$', r'$\rho=0.001$', r'$\rho=0.002$', r'$\rho=0.004$', r'$\rho=0.008$', r'$\rho=0.01$']
    
legend_names = []

output_filename = 'horizontal_legend.png'

utility.create_horizontal_legend(legend_names, output_filename)