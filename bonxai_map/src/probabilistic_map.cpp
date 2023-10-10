#include "bonxai_map/probabilistic_map.hpp"
#include <eigen3/Eigen/Geometry>
#include <unordered_set>

namespace Bonxai
{

const int32_t ProbabilisticMap::UnknownProbability = ProbabilisticMap::logods(0.5f);



ProbabilisticMap::OpenVdbGrid& ProbabilisticMap::grid()
{
  return _grid;
}

ProbabilisticMap::ProbabilisticMap(double resolution)
  : _grid()
  , _accessor(_grid.getAccessor())
{
  _grid.setTransform(openvdb::math::Transform::createLinearTransform(resolution));
  _grid.setGridClass(openvdb::GRID_LEVEL_SET);
  _accessor = _grid.getAccessor();
}

const ProbabilisticMap::OpenVdbGrid& ProbabilisticMap::grid() const
{
  return _grid;
}

const ProbabilisticMap::Options& ProbabilisticMap::options() const
{
  return _options;
}

void ProbabilisticMap::setOptions(const Options& options)
{
  _options = options;
}

void ProbabilisticMap::addHitPoint(const Vector3D &point)
{
  const auto coord = openvdb::Coord::floor(_grid.worldToIndex(point.data()));
  int32_t cell_val = _accessor.getValue(coord);
  CellT* cell = reinterpret_cast<CellT*>(&cell_val);

  if (cell->update_id != _update_count)
  {
    cell->probability_log = std::min(cell->probability_log + _options.prob_hit_log,
                                     _options.clamp_max_log);

    cell->update_id = _update_count;
    _accessor.setValue(coord, cell_val);
    _hit_coords.push_back(coord);
  }
}

void ProbabilisticMap::addMissPoint(const Vector3D &point)
{
  const auto coord = openvdb::Coord::floor(_grid.worldToIndex(point.data()));
  int32_t cell_val = _accessor.getValue(coord);
  CellT* cell = reinterpret_cast<CellT*>(&cell_val);

  if (cell->update_id != _update_count)
  {
    cell->probability_log = std::max(cell->probability_log + _options.prob_miss_log,
                                     _options.clamp_min_log);

    cell->update_id = _update_count;
    _accessor.setValue(coord, cell_val);
    _miss_coords.push_back(coord);
  }
}

bool ProbabilisticMap::isOccupied(const openvdb::Coord &coord) const
{
  int32_t cell_val = _accessor.getValue(coord);
  CellT* cell = reinterpret_cast<CellT*>(&cell_val);
  return cell->probability_log > _options.occupancy_threshold_log;
}

bool ProbabilisticMap::isUnknown(const openvdb::Coord &coord) const
{
  int32_t cell_val = _accessor.getValue(coord);
  CellT* cell = reinterpret_cast<CellT*>(&cell_val);
  return cell->probability_log == _options.occupancy_threshold_log;
}

bool ProbabilisticMap::isFree(const openvdb::Coord &coord) const
{
  int32_t cell_val = _accessor.getValue(coord);
  CellT* cell = reinterpret_cast<CellT*>(&cell_val);
  return cell->probability_log < _options.occupancy_threshold_log;
}

void Bonxai::ProbabilisticMap::updateFreeCells(const Vector3D& origin)
{
  auto accessor = _grid.getAccessor();

  // same as addMissPoint, but using lambda will force inlining
  auto clearPoint = [this, &accessor](const openvdb::Coord& coord)
  {
    int32_t cell_val = accessor.getValue(coord);
    CellT* cell = reinterpret_cast<CellT*>(&cell_val);

    if (cell->update_id != _update_count)
    {
      cell->probability_log = std::max(cell->probability_log + _options.prob_miss_log,
          _options.clamp_min_log);

      cell->update_id = _update_count;
      accessor.setValue(coord, cell_val);
    }
    return true;
  };

  const auto coord_origin = openvdb::Coord::floor(_grid.worldToIndex(origin.data()));

  for (const auto& coord_end : _hit_coords)
  {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  _hit_coords.clear();

  for (const auto& coord_end : _miss_coords)
  {
    RayIterator(coord_origin, coord_end, clearPoint);
  }
  _miss_coords.clear();

  if (++_update_count == 4)
  {
    _update_count = 1;
  }
}

void ProbabilisticMap::getOccupiedVoxels(std::vector<openvdb::Coord>& coords)
{
  coords.clear();
  for (auto cell_it = _grid.cbeginValueOn(); cell_it.test(); cell_it.next()) {
    int32_t cell_val = cell_it.getValue();
    CellT* cell = reinterpret_cast<CellT*>(&cell_val);
    if (cell->probability_log > _options.occupancy_threshold_log)
    {
      coords.push_back(cell_it.getCoord());
    }
  }
}

void ProbabilisticMap::getFreeVoxels(std::vector<openvdb::Coord>& coords)
{
  coords.clear();
  for (auto cell_it = _grid.cbeginValueOn(); cell_it.test(); cell_it.next()) {
    int32_t cell_val = cell_it.getValue();
    CellT* cell = reinterpret_cast<CellT*>(&cell_val);
    if (cell->probability_log < _options.occupancy_threshold_log)
    {
      coords.push_back(cell_it.getCoord());
    }
  }
}

}  // namespace Bonxai
