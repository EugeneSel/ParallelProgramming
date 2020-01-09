#ifndef _DISPLAY_HPP_
# define _DISPLAY_HPP_
#include <utility>
# include <vector>
#include "fractal_land.hpp"
#include "pheromone.hpp"
#include "ant.hpp"
# include "gui/window.hpp"
# include "gui/point.hpp"

struct display_t
{
    display_t( const fractal_land& land, const pheromone& phen, 
               const position_t& pos_nest, const position_t& pos_food,
               const std::vector<ant>& ants, gui::window& win );
    display_t(const display_t& ) = delete;
    ~display_t()=default;

    void display( const std::size_t& compteur );
private:
    struct graph_context;
    const fractal_land& m_ref_land;
    const pheromone& m_ref_phen;
    const position_t& m_pos_nest;
    const position_t& m_pos_food;
    const std::vector<ant>& m_ref_ants;
    std::vector<std::size_t> m_curve;    
    gui::window& m_ref_win;
    // La carte territoriale à afficher :
    std::vector<gui::point> land_points;
};


#endif