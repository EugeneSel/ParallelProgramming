# include "fractal_land.hpp"

void 
fractal_land::compute_subgrid( int log_subgrid_dim, int iB, int jB, double deviation,
                               std::mt19937& gen )
{
    // Génère des réels pseudo-aléatoires compris dans [-deviation;+deviation]
    std::uniform_real_distribution<> distrib_dev(-deviation,deviation);
    auto random_dev = [&gen,&distrib_dev] () { return distrib_dev(gen); };

    fractal_land& cur_land = *this;
    unsigned long dim_ss_grid = 1UL<<(log_subgrid_dim);
    unsigned long iBeg = iB*dim_ss_grid;
    unsigned long jBeg = jB*dim_ss_grid;
    int mid_ind = dim_ss_grid/2;
    int i_mid = iBeg+mid_ind, j_mid = jBeg+mid_ind;
    int iEnd  = iBeg + dim_ss_grid, jEnd = jBeg + dim_ss_grid;
    cur_land(i_mid,jBeg)  = 0.5* (cur_land(iBeg,jBeg)+cur_land(iEnd,jBeg))+mid_ind*random_dev();
    cur_land(iBeg,j_mid)  = 0.5* (cur_land(iBeg,jBeg)+cur_land(iBeg,jEnd))+mid_ind*random_dev();
    cur_land(i_mid,jEnd)  = 0.5* (cur_land(iBeg,jEnd)+cur_land(iEnd,jEnd))+mid_ind*random_dev();
    cur_land(iEnd,j_mid)  = 0.5* (cur_land(iEnd,jBeg)+cur_land(iEnd,jEnd))+mid_ind*random_dev();
    cur_land(i_mid,j_mid) = 0.25*(cur_land(i_mid,jBeg)+cur_land(iBeg,j_mid)+cur_land(i_mid,jEnd)+cur_land(iEnd,j_mid))+ mid_ind*random_dev();
}


fractal_land::fractal_land( const dim_t& ln2_dim, unsigned long nbSeeds, double deviation, int seed ) :
    m_dimensions(0), m_altitude()
{
    // dim_ss_grid = 2^{ln2_dim}
    unsigned long dim_ss_grid = 1UL<<(ln2_dim);
    m_dimensions = nbSeeds*dim_ss_grid+1;
    container(m_dimensions*m_dimensions).swap(m_altitude);

    std::random_device rd;
    std::mt19937 gen(rd());

    // Seed the engine with an unsigned int
    gen.seed(seed);
    // Génère des réels pseudo-aléatoires compris dans [-deviation;+deviation]
    std::uniform_real_distribution<> distrib_alt(0.,dim_ss_grid*deviation);

    auto random_alt = [&gen,&distrib_alt] () { return distrib_alt(gen); };

    fractal_land& cur_land = *this;
    // Calcul des points initiaux :
    for ( dim_t i = 0; i < m_dimensions; i += dim_ss_grid )
        for ( dim_t j = 0; j < m_dimensions; j += dim_ss_grid )
            cur_land(i,j) = random_alt();
    // Puis on itère pour calculer le paysage fractal :
    dim_t ldim = ln2_dim;
    while (ldim > 1)
    {
        ldim -= 1;
        dim_ss_grid /= 2;
        nbSeeds *= 2;
        for ( unsigned long iB = 0; iB < nbSeeds; ++iB )
            for ( unsigned long jB = 0; jB < nbSeeds; ++jB )
               compute_subgrid( ldim, iB, jB, deviation, gen);                
    }
}