// The regrouping of one block along one step of a chain: the runs of entries a step adds, and the
// plans that exchange two codomain legs or bend a leg from one side onto the other.
#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include "layout.h"

namespace sym
{

  /**
   * @brief One run of entries of a regrouping.
   * @details The entry `first + at * from_step` of the block a shift reads is added to the entry
   * `last + at * to_step` of the block it writes, scaled by `coefficient`.  A block is a contiguous
   * matrix, so a run along the columns has step one on both sides, while a run along the rows
   * carries the row stride.
   */
  struct Piece
  {
    int64_t first;
    int64_t last;
    int64_t from_step;
    int64_t to_step;
    int64_t run;
    std::complex<double> coefficient;
  };

  /**
   * @brief A shift of one block of a tensor along one step of a chain.
   * @details `pieces` add the block `from` of the tensor into the block `to`, each streamed through
   * a dense buffer of `rows x columns`; the two shapes are recorded here so that a step need not look
   * the layout up again for every block.
   */
  struct Shift
  {
    BlockKey from;
    BlockKey to;
    int64_t rows;
    int64_t columns;
    std::vector<Piece> pieces;
  };

  /**
   * @brief The regroupings of one tensor through one step.
   * @details Held together so that a cached plan stays alive as one object: the shifts of the
   * tensor's blocks.
   */
  struct Regrouping
  {
    std::vector<Shift> shifts;
  };

  /**
   * @brief The regrouping of exchanging the codomain legs `p` and `p + 1` of one block.
   * @details The trees that agree outside the pair recouple by F and R, and the width the two legs
   * carry is read at the exchanged position on the other side.
   * @param cat The category.
   * @param codomain The codomain legs before the exchange.
   * @param domain The domain legs, which the step leaves alone.
   * @param key The block the step acts on.
   * @param qtotal The total charge of the tensor.
   * @param p The position of the pair in the codomain.
   * @param layouts Where the layouts of the sides are read from.
   */
  Regrouping exchange_regrouping(const std::shared_ptr<const FusionCategory> &cat,
                                 const std::vector<Leg> &codomain,
                                 const std::vector<Leg> &domain,
                                 const BlockKey &key,
                                 const Sector &qtotal,
                                 int64_t p,
                                 const LayoutSource &layouts);

  /**
   * @brief The regrouping of bending the last leg of one side onto the other, for one block.
   * @details The giving tree keeps everything but its last vertex, the taking tree gains one, and
   * the recoupling of the two vertices carries the copies of the moved leg into the columns of the
   * taking side.
   * @param cat The category.
   * @param codomain The codomain legs before the step.
   * @param domain The domain legs before the step.
   * @param key The block the step acts on.
   * @param qtotal The total charge of the tensor.
   * @param from_codomain Whether the codomain gives the leg (`true`) or takes it.
   * @param layouts Where the layouts of the sides are read from.
   */
  Regrouping bend_regrouping(const std::shared_ptr<const FusionCategory> &cat,
                             const std::vector<Leg> &codomain,
                             const std::vector<Leg> &domain,
                             const BlockKey &key,
                             const Sector &qtotal,
                             bool from_codomain,
                             const LayoutSource &layouts);

} // namespace sym
