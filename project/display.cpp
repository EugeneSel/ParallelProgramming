#include "display.hpp"
#include <iostream>
#include "gui/colors.hpp"
#include "gui/point.hpp"
#include "gui/quad.hpp"
#include "gui/segment.hpp"
#include "gui/triangle.hpp"

display_t::display_t( const fractal_land& land, const pheromone& phen, const position_t& pos_nest,
                      const position_t& pos_food, const std::vector< ant >& ants, gui::window& win )
    : m_ref_land(land),
      m_ref_phen(phen),
      m_pos_nest(pos_nest),
      m_pos_food(pos_food),
      m_ref_ants(ants),
      m_curve( ),
      m_ref_win( win ),
      land_points( ) {
    m_curve.reserve( 50000 );
    land_points.reserve( 512 * 522 );
    // Affichage premier cadran et second cadran du paysage
    for ( fractal_land::dim_t i = 0; i < m_ref_land.dimensions( ); ++i )
        for ( fractal_land::dim_t j = 0; j < m_ref_land.dimensions( ); ++j ) {
            double c = m_ref_land( i, j );
            if ( ( i != pos_food.first) or ( j != pos_food.second ) )
                land_points.push_back( gui::point( i, j, 0, gui::color::rgba( {c, c, c} ) ) );
            else
                land_points.push_back( gui::point( i, j, 0, gui::color::rgba( {255, 0, 0} ) ) );
            /*land_points.push_back(
                gui::point( i + m_ref_land.dimensions( ) + 10, j, 0, gui::color::rgba( {c / 4, c / 4, c / 4} ) ) );*/
        }
}

void display_t::display( const std::size_t& compteur ) {
    m_ref_win.clear( );
    // Affichage des paysages
    gui::draw( land_points );
    // Affichage des fourmis dans cadran en haut à gauche :
    std::vector< gui::point > m_p_ants;
    m_p_ants.reserve( m_ref_ants.size( ) );
    for ( auto& ant : m_ref_ants ) {
        const position_t& pos_ant = ant.get_position( );
        m_p_ants.push_back( gui::point( pos_ant.first, pos_ant.second, 0, gui::color::rgba( {0, 255, 255, 128} ) ) );
    }
    gui::draw(m_p_ants);
    // Affichage des phéronomes dans le cadran en haut à droite :
    std::vector< gui::point > m_p_phen;
    m_p_phen.reserve( m_ref_land.dimensions( ) * m_ref_land.dimensions( ) );
    for ( fractal_land::dim_t i = 0; i < m_ref_land.dimensions( ); ++i )
        for ( fractal_land::dim_t j = 0; j < m_ref_land.dimensions( ); ++j ) {
            m_p_phen.push_back( gui::point(
                i+ m_ref_land.dimensions( ) + 10, j, 0, gui::color::rgba( {std::min( 1., (double)m_ref_phen( i, j )[0] ),
                    std::min((double)m_ref_phen( i, j )[1], 1.),  0.} ) ) );
        }
    gui::draw(m_p_phen);
    // Affichage de la courbe d'enfouragement :
    m_curve.push_back(compteur);
    if ( m_curve.size( ) > 1 ) {
        std::vector< gui::segment > m_p_curve;
	 int sz_win = m_ref_win.size( ).first;
	int ydec = m_ref_win.size( ).second - 1;
        m_p_curve.reserve( sz_win );
        double h_max_val = 256. / std::max(double( m_curve.back( )), 1.);
        double step      = double(sz_win) / (double)( m_curve.size( ) );
        if ( step > 1. ) {
            for ( std::size_t i = 0; i < m_curve.size( ) - 1; i++ ) {
                m_p_curve.push_back(
                    {gui::point( i * step, ydec - m_curve[i] * h_max_val, 0., gui::color::rgba( {255, 255, 127} ) ),
                     gui::point( ( i + 1 ) * step, ydec - m_curve[i + 1] * h_max_val, 0.,
                                 gui::color::rgba( {255, 255, 127} ) )} );
            }
        } else {
            double inv_step = 1. / (step+0.001);
            for ( int i = 0; i < sz_win; ++i ) {
                int ind  = int( i * inv_step);
                int ind2 = int( ( i + 1 ) * inv_step );
                m_p_curve.push_back(
                    {gui::point( i, ydec - m_curve[ind] * h_max_val, 0., gui::color::rgba( {255, 255, 127} ) ),
                     gui::point( i + 1, ydec - m_curve[ind2] * h_max_val, 0.,
                                 gui::color::rgba( {255, 255, 127} ) )} );
            }
        }
        gui::draw(m_p_curve);
    }
}
