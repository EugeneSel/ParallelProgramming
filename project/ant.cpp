#include "ant.hpp"
#include <iostream>
#include <random>

double ant::m_eps = 0.;

void ant::advance( pheromone& phen, const fractal_land& land, const position_t& pos_food, const position_t& pos_nest,
                   std::size_t& cpteur_food ) 
{
    std::random_device                       rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937                             gen( rd( ) );  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution< double > ant_choice( 0., 1. );
    std::uniform_int_distribution<int> dir_choice(1,4);
    double                                   consumed_time = 0.;
    // Tant que la fourmi peut encore bouger dans le pas de temps imparti
    while ( consumed_time < 1. ) {
        // Si la fourmi est chargée, elle suit les phéromones de deuxième type, sinon ceux du premier.
        int        ind_pher    = ( is_loaded( ) ? 1 : 0 );
        double     choix       = ant_choice( gen );
        position_t old_pos_ant = get_position( );
        position_t new_pos_ant = old_pos_ant;
        double max_phen    = std::max( {phen( new_pos_ant.first - 1, new_pos_ant.second )[ind_pher],
                                     phen( new_pos_ant.first + 1, new_pos_ant.second )[ind_pher],
                                     phen( new_pos_ant.first, new_pos_ant.second - 1 )[ind_pher],
                                     phen( new_pos_ant.first, new_pos_ant.second + 1 )[ind_pher]} );
        if ( ( choix > m_eps ) || ( max_phen <= 0. ) ) {
            do {
                new_pos_ant = old_pos_ant;
                int d = dir_choice(gen);
                if ( d==1 ) new_pos_ant.first  -= 1;
                if ( d==2 ) new_pos_ant.second -= 1;
                if ( d==3 ) new_pos_ant.first  += 1;
                if ( d==4 ) new_pos_ant.second += 1;

            } while ( phen[new_pos_ant][ind_pher] == -1 );
        } else {
            // On choisit la case où le phéromone est le plus fort.
            if ( phen( new_pos_ant.first - 1, new_pos_ant.second )[ind_pher] == max_phen )
                new_pos_ant.first -= 1;
            else if ( phen( new_pos_ant.first + 1, new_pos_ant.second )[ind_pher] == max_phen )
                new_pos_ant.first += 1;
            else if ( phen( new_pos_ant.first, new_pos_ant.second - 1 )[ind_pher] == max_phen )
                new_pos_ant.second -= 1;
            else  // if (phen(new_pos_ant.first,new_pos_ant.second+1)[ind_pher] == max_phen)
                new_pos_ant.second += 1;
        }
        consumed_time += land( new_pos_ant.first, new_pos_ant.second);
        phen.mark_pheromone( new_pos_ant );
        m_position = new_pos_ant;
        if (get_position() == pos_nest) {
            if (is_loaded()) {
                cpteur_food += 1;
            }
            unset_loaded( );
        }
        if (get_position( ) == pos_food) {
            set_loaded( );
        }
    }
}

void ant::set_position(const double& first, const double& second) {
    m_position.first = first;
    m_position.second = second;
}