#ifndef _PHERONOME_HPP_
#define _PHERONOME_HPP_
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include "basic_types.hpp"

using namespace std;

/**
 * @brief Carte des phéronomes
 * @details Gère une carte des phéronomes avec leurs mis à jour ( dont l'évaporation )
 *
 */
class pheromone {
public:
    using size_t      = unsigned long;
    using pheromone_t = std::array< double, 2 >;

    /**
     * @brief Construit une carte initiale des phéronomes
     * @details La carte des phéronomes est initialisées à zéro ( neutre )
     *          sauf pour les bords qui sont marqués comme indésirables
     *
     * @param dim Nombre de cellule dans chaque direction
     * @param alpha Paramètre de bruit
     * @param beta Paramêtre d'évaporation
     */
    pheromone( size_t dim, const position_t& pos_food, const position_t& pos_nest,
               double alpha = 0.7, double beta = 0.9999 )
        : m_dim( dim ),
          m_stride( dim + 2 ),
          m_alpha(alpha), m_beta(beta),
          m_map_of_pheromone( m_stride * m_stride, {{0., 0.}} ),
          m_buffer_pheromone( ),
          m_pos_nest( pos_nest ),
          m_pos_food( pos_food ) 
          {
        m_map_of_pheromone[index(pos_food)][0] = 1.;
        m_map_of_pheromone[index(pos_nest)][1] = 1.;
        cl_update( );
        m_buffer_pheromone = m_map_of_pheromone;
    }
    pheromone( const pheromone& ) = delete;
    pheromone( pheromone&& )      = delete;
    ~pheromone( )                 = default;

    pheromone_t& operator( )( size_t i, size_t j ) {
        return m_map_of_pheromone[( i + 1 ) * m_stride + ( j + 1 )];
    }

    const pheromone_t& operator( )( size_t i, size_t j ) const {
        return m_map_of_pheromone[( i + 1 ) * m_stride + ( j + 1 )];
    }

    pheromone_t& operator[] ( const position_t& pos ) {
      return m_map_of_pheromone[index(pos)];
    }

    const pheromone_t& operator[] ( const position_t& pos ) const {
      return m_map_of_pheromone[index(pos)];
    }

    const vector<pheromone_t> get_m_map_of_pheromone() {
        return m_map_of_pheromone;
    } 

    void do_evaporation( ) {
        for ( std::size_t i = 1; i <= m_dim; ++i )
            for ( std::size_t j = 1; j <= m_dim; ++j ) {
                m_buffer_pheromone[i * m_stride + j][0] *= m_beta;
                m_buffer_pheromone[i * m_stride + j][1] *= m_beta;
            }
    }

    void mark_pheromone( const position_t& pos ) {
      std::size_t i = pos.first;
      std::size_t j = pos.second;
        assert( i >= 0 );
        assert( j >= 0 );
        assert( i < m_dim );
        assert( j < m_dim );
        pheromone&         phen        = *this;
        const pheromone_t& left_cell   = phen( i - 1, j );
        const pheromone_t& right_cell  = phen( i + 1, j );
        const pheromone_t& upper_cell  = phen( i, j - 1 );
        const pheromone_t& bottom_cell = phen( i, j + 1 );
        double             v1_left     = std::max( left_cell[0], 0. );
        double             v2_left     = std::max( left_cell[1], 0. );
        double             v1_right    = std::max( right_cell[0], 0. );
        double             v2_right    = std::max( right_cell[1], 0. );
        double             v1_upper    = std::max( upper_cell[0], 0. );
        double             v2_upper    = std::max( upper_cell[1], 0. );
        double             v1_bottom   = std::max( bottom_cell[0], 0. );
        double             v2_bottom   = std::max( bottom_cell[1], 0. );
        m_buffer_pheromone[( i + 1 ) * m_stride + ( j + 1 )][0] =
            m_alpha * std::max( {v1_left, v1_right, v1_upper, v1_bottom} ) +
            ( 1 - m_alpha ) * 0.25 * ( v1_left + v1_right + v1_upper + v1_bottom );
        m_buffer_pheromone[( i + 1 ) * m_stride + ( j + 1 )][1] =
            m_alpha * std::max( {v2_left, v2_right, v2_upper, v2_bottom} ) +
            ( 1 - m_alpha ) * 0.25 * ( v2_left + v2_right + v2_upper + v2_bottom );
    }

    void update( ) {
        m_map_of_pheromone.swap(m_buffer_pheromone);
        cl_update( );
        m_map_of_pheromone[( m_pos_food.first + 1 ) * m_stride + m_pos_food.second + 1][0] = 1;
        m_map_of_pheromone[( m_pos_nest.first + 1 ) * m_stride + m_pos_nest.second + 1][1] = 1;
    }

    void update_map(const vector<double>& new_map) {
        unsigned k = 0;

        for (size_t i = 1; i <= m_dim; ++i)
            for (size_t j = 1; j <= m_dim; ++j) {
                m_map_of_pheromone[i * m_stride + j][0] = new_map[k];
                m_map_of_pheromone[i * m_stride + j][1] = new_map[k + 1];

                k += 2;
            } 
    }

private:
    size_t index( const position_t& pos ) const
    {
      return (pos.first+1)*m_stride + pos.second + 1;
    }
    /**
     * @brief Mets à jour les conditions limites sur les cellules fantômes
     * @details Mets à jour les conditions limites sur les cellules fantômes :
     *     pour l'instant, on se contente simplement de mettre ces cellules avec
     *     des valeurs à -1 pour être sûr que les fourmis évitent ces cellules
     */
    void cl_update( ) {
        // On mets tous les bords à -1 pour les marquer comme indésirables :
        for ( unsigned long j = 0; j < m_stride; ++j ) {
            m_map_of_pheromone[j]                            = {{-1., -1.}};
            m_map_of_pheromone[j + m_stride * ( m_dim + 1 )] = {{-1., -1.}};
            m_map_of_pheromone[j * m_stride]                 = {{-1., -1.}};
            m_map_of_pheromone[j * m_stride + m_dim + 1]     = {{-1., -1.}};
        }
    }
    unsigned long              m_dim, m_stride;
    double                     m_alpha, m_beta;
    std::vector< pheromone_t > m_map_of_pheromone, m_buffer_pheromone;
    position_t m_pos_nest, m_pos_food;
};

#endif