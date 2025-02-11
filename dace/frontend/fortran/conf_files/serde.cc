#ifndef __DACE_SERDE__
#define __DACE_SERDE__

#include <cassert>
#include <iostream>
#include <istream>
#include <sstream>

#include "radiation.h"

namespace serde {
struct array_meta {
  int rank = 0;
  std::vector<int> size, lbound;

  int volume() const {
    return std::reduce(size.begin(), size.end(), 1, std::multiplies<int>());
  }
};

std::string scroll_space(std::istream& s) {
  std::string out;
  while (!s.eof() && (!s.peek() || isspace(s.peek()))) {
    out += s.get();
    assert(s.good());
  }
  return out;
}

std::string read_line(std::istream& s,
                      const std::optional<std::string>& should_contain = {}) {
  if (s.eof()) return "<eof>";
  scroll_space(s);
  char bin[101];
  s.getline(bin, 100);
  assert(s.good());
  if (should_contain) {
    bool ok = (std::string(bin).find(*should_contain) != std::string::npos);
    if (!ok) {
      std::cerr << "Expected: '" << *should_contain << "'; got: '" << bin << "'"
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  return {bin};
}

template <typename T>
void read_scalar(T& x, std::istream& s) {
  if (s.eof()) return;
  scroll_space(s);
  s >> x;
}

void read_scalar(float& x, std::istream& s) {
  if (s.eof()) return;
  scroll_space(s);
  long double y;
  s >> y;
  x = y;
}

void read_scalar(double& x, std::istream& s) {
  if (s.eof()) return;
  scroll_space(s);
  long double y;
  s >> y;
  x = y;
}

void read_scalar(bool& x, std::istream& s) {
  char c;
  read_scalar(c, s);
  assert(c == '1' or c == '0');
  x = (c == '1');
}

array_meta read_array_meta(std::istream& s) {
  array_meta m;
  read_line(s, {"# rank"});  // Should contain '# rank'
  read_scalar(m.rank, s);
  m.size.resize(m.rank);
  m.lbound.resize(m.rank);
  read_line(s, {"# size"});  // Should contain '# size'
  for (int i = 0; i < m.rank; ++i) {
    read_scalar(m.size[i], s);
  }
  read_line(s, {"# lbound"});  // Should contain '# lbound'
  for (int i = 0; i < m.rank; ++i) {
    read_scalar(m.lbound[i], s);
  }
  return m;
}

void deserialize(float* x, std::istream& s) { read_scalar(*x, s); }
void deserialize(double* x, std::istream& s) { read_scalar(*x, s); }
void deserialize(long double* x, std::istream& s) { read_scalar(*x, s); }
void deserialize(int* x, std::istream& s) { read_scalar(*x, s); }
void deserialize(long* x, std::istream& s) { read_scalar(*x, s); }
void deserialize(long long* x, std::istream& s) { read_scalar(*x, s); }
void deserialize(bool* x, std::istream& s) { read_scalar(*x, s); }

void deserialize(aerosol_type* x, std::istream& s) {
  bool yep;
  array_meta m;
  read_line(s, {"# od_sw"});  // Should contain '# od_sw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_od_sw_d_0_s_61 = m.size[0];
    x->__f2dace_SA_od_sw_d_1_s_62 = m.size[1];
    x->__f2dace_SA_od_sw_d_2_s_63 = m.size[2];
    x->__f2dace_SOA_od_sw_d_0_s_61 = m.lbound[0];
    x->__f2dace_SOA_od_sw_d_1_s_62 = m.lbound[1];
    x->__f2dace_SOA_od_sw_d_2_s_63 = m.lbound[2];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->od_sw = new std::remove_pointer<decltype(x->od_sw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->od_sw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# ssa_sw"});  // Should contain '# ssa_sw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_ssa_sw_d_0_s_64 = m.size[0];
    x->__f2dace_SA_ssa_sw_d_1_s_65 = m.size[1];
    x->__f2dace_SA_ssa_sw_d_2_s_66 = m.size[2];
    x->__f2dace_SOA_ssa_sw_d_0_s_64 = m.lbound[0];
    x->__f2dace_SOA_ssa_sw_d_1_s_65 = m.lbound[1];
    x->__f2dace_SOA_ssa_sw_d_2_s_66 = m.lbound[2];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->ssa_sw = new std::remove_pointer<decltype(x->ssa_sw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->ssa_sw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# g_sw"});  // Should contain '# g_sw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_g_sw_d_0_s_67 = m.size[0];
    x->__f2dace_SA_g_sw_d_1_s_68 = m.size[1];
    x->__f2dace_SA_g_sw_d_2_s_69 = m.size[2];
    x->__f2dace_SOA_g_sw_d_0_s_67 = m.lbound[0];
    x->__f2dace_SOA_g_sw_d_1_s_68 = m.lbound[1];
    x->__f2dace_SOA_g_sw_d_2_s_69 = m.lbound[2];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->g_sw = new std::remove_pointer<decltype(x->g_sw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->g_sw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# od_lw"});  // Should contain '# od_lw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_od_lw_d_0_s_70 = m.size[0];
    x->__f2dace_SA_od_lw_d_1_s_71 = m.size[1];
    x->__f2dace_SA_od_lw_d_2_s_72 = m.size[2];
    x->__f2dace_SOA_od_lw_d_0_s_70 = m.lbound[0];
    x->__f2dace_SOA_od_lw_d_1_s_71 = m.lbound[1];
    x->__f2dace_SOA_od_lw_d_2_s_72 = m.lbound[2];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->od_lw = new std::remove_pointer<decltype(x->od_lw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->od_lw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# ssa_lw"});  // Should contain '# ssa_lw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_ssa_lw_d_0_s_73 = m.size[0];
    x->__f2dace_SA_ssa_lw_d_1_s_74 = m.size[1];
    x->__f2dace_SA_ssa_lw_d_2_s_75 = m.size[2];
    x->__f2dace_SOA_ssa_lw_d_0_s_73 = m.lbound[0];
    x->__f2dace_SOA_ssa_lw_d_1_s_74 = m.lbound[1];
    x->__f2dace_SOA_ssa_lw_d_2_s_75 = m.lbound[2];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->ssa_lw = new std::remove_pointer<decltype(x->ssa_lw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->ssa_lw[i]), s);
    }

  }  // CONCLUDING IF
}

void deserialize(cloud_type* x, std::istream& s) {
  bool yep;
  array_meta m;
  read_line(s, {"# mixing_ratio"});  // Should contain '# mixing_ratio'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_mixing_ratio_d_0_s_143 = m.size[0];
    x->__f2dace_SA_mixing_ratio_d_1_s_144 = m.size[1];
    x->__f2dace_SA_mixing_ratio_d_2_s_145 = m.size[2];
    x->__f2dace_SOA_mixing_ratio_d_0_s_143 = m.lbound[0];
    x->__f2dace_SOA_mixing_ratio_d_1_s_144 = m.lbound[1];
    x->__f2dace_SOA_mixing_ratio_d_2_s_145 = m.lbound[2];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->mixing_ratio =
        new std::remove_pointer<decltype(x->mixing_ratio)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->mixing_ratio[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# q_liq"});  // Should contain '# q_liq'

  read_line(s, {"# assoc"});  // Should contain '# assoc'
  deserialize(&yep, s);

  read_line(s, {"=>"});  // Should contain '=> ...'
  x->q_liq = nullptr;

  read_line(s, {"# q_ice"});  // Should contain '# q_ice'

  read_line(s, {"# assoc"});  // Should contain '# assoc'
  deserialize(&yep, s);

  read_line(s, {"=>"});  // Should contain '=> ...'
  x->q_ice = nullptr;

  read_line(s, {"# re_liq"});  // Should contain '# re_liq'

  read_line(s, {"# assoc"});  // Should contain '# assoc'
  deserialize(&yep, s);

  read_line(s, {"=>"});  // Should contain '=> ...'
  x->re_liq = nullptr;

  read_line(s, {"# re_ice"});  // Should contain '# re_ice'

  read_line(s, {"# assoc"});  // Should contain '# assoc'
  deserialize(&yep, s);

  read_line(s, {"=>"});  // Should contain '=> ...'
  x->re_ice = nullptr;

  read_line(s, {"# fraction"});  // Should contain '# fraction'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_fraction_d_0_s_154 = m.size[0];
    x->__f2dace_SA_fraction_d_1_s_155 = m.size[1];
    x->__f2dace_SOA_fraction_d_0_s_154 = m.lbound[0];
    x->__f2dace_SOA_fraction_d_1_s_155 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->fraction =
        new std::remove_pointer<decltype(x->fraction)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->fraction[i]), s);
    }

  }  // CONCLUDING IF
}

void deserialize(cloud_optics_type* x, std::istream& s) {
  bool yep;
  array_meta m;
  read_line(s, {"# liq_coeff_lw"});  // Should contain '# liq_coeff_lw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_liq_coeff_lw_d_0_s_28 = m.size[0];
    x->__f2dace_SA_liq_coeff_lw_d_1_s_29 = m.size[1];
    x->__f2dace_SOA_liq_coeff_lw_d_0_s_28 = m.lbound[0];
    x->__f2dace_SOA_liq_coeff_lw_d_1_s_29 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->liq_coeff_lw =
        new std::remove_pointer<decltype(x->liq_coeff_lw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->liq_coeff_lw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# liq_coeff_sw"});  // Should contain '# liq_coeff_sw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_liq_coeff_sw_d_0_s_30 = m.size[0];
    x->__f2dace_SA_liq_coeff_sw_d_1_s_31 = m.size[1];
    x->__f2dace_SOA_liq_coeff_sw_d_0_s_30 = m.lbound[0];
    x->__f2dace_SOA_liq_coeff_sw_d_1_s_31 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->liq_coeff_sw =
        new std::remove_pointer<decltype(x->liq_coeff_sw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->liq_coeff_sw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# ice_coeff_lw"});  // Should contain '# ice_coeff_lw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_ice_coeff_lw_d_0_s_32 = m.size[0];
    x->__f2dace_SA_ice_coeff_lw_d_1_s_33 = m.size[1];
    x->__f2dace_SOA_ice_coeff_lw_d_0_s_32 = m.lbound[0];
    x->__f2dace_SOA_ice_coeff_lw_d_1_s_33 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->ice_coeff_lw =
        new std::remove_pointer<decltype(x->ice_coeff_lw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->ice_coeff_lw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# ice_coeff_sw"});  // Should contain '# ice_coeff_sw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_ice_coeff_sw_d_0_s_34 = m.size[0];
    x->__f2dace_SA_ice_coeff_sw_d_1_s_35 = m.size[1];
    x->__f2dace_SOA_ice_coeff_sw_d_0_s_34 = m.lbound[0];
    x->__f2dace_SOA_ice_coeff_sw_d_1_s_35 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->ice_coeff_sw =
        new std::remove_pointer<decltype(x->ice_coeff_sw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->ice_coeff_sw[i]), s);
    }

  }  // CONCLUDING IF
}

void deserialize(gas_type* x, std::istream& s) {
  bool yep;
  array_meta m;
}

void deserialize(config_type* x, std::istream& s) {
  bool yep;
  array_meta m;
  read_line(
      s,
      {"# i_emiss_from_band_lw"});  // Should contain '# i_emiss_from_band_lw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_i_emiss_from_band_lw_d_0_s_41 = m.size[0];
    x->__f2dace_SOA_i_emiss_from_band_lw_d_0_s_41 = m.lbound[0];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->i_emiss_from_band_lw = new std::remove_pointer<
        decltype(x->i_emiss_from_band_lw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->i_emiss_from_band_lw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s,
            {"# sw_albedo_weights"});  // Should contain '# sw_albedo_weights'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_albedo_weights_d_0_s_42 = m.size[0];
    x->__f2dace_SA_sw_albedo_weights_d_1_s_43 = m.size[1];
    x->__f2dace_SOA_sw_albedo_weights_d_0_s_42 = m.lbound[0];
    x->__f2dace_SOA_sw_albedo_weights_d_1_s_43 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_albedo_weights = new std::remove_pointer<
        decltype(x->sw_albedo_weights)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_albedo_weights[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s,
            {"# i_band_from_reordered_g_lw"});  // Should contain '#
                                                // i_band_from_reordered_g_lw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_i_band_from_reordered_g_lw_d_0_s_44 = m.size[0];
    x->__f2dace_SOA_i_band_from_reordered_g_lw_d_0_s_44 = m.lbound[0];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->i_band_from_reordered_g_lw = new std::remove_pointer<
        decltype(x->i_band_from_reordered_g_lw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->i_band_from_reordered_g_lw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s,
            {"# i_band_from_reordered_g_sw"});  // Should contain '#
                                                // i_band_from_reordered_g_sw'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_i_band_from_reordered_g_sw_d_0_s_45 = m.size[0];
    x->__f2dace_SOA_i_band_from_reordered_g_sw_d_0_s_45 = m.lbound[0];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->i_band_from_reordered_g_sw = new std::remove_pointer<
        decltype(x->i_band_from_reordered_g_sw)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->i_band_from_reordered_g_sw[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# cloud_optics"});  // Should contain '# cloud_optics'

  x->cloud_optics = new std::remove_pointer<decltype(x->cloud_optics)>::type;
  deserialize(x->cloud_optics, s);
}

void deserialize(flux_type* x, std::istream& s) {
  bool yep;
  array_meta m;
  read_line(
      s,
      {"# sw_dn_diffuse_surf_g"});  // Should contain '# sw_dn_diffuse_surf_g'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_dn_diffuse_surf_g_d_0_s_92 = m.size[0];
    x->__f2dace_SA_sw_dn_diffuse_surf_g_d_1_s_93 = m.size[1];
    x->__f2dace_SOA_sw_dn_diffuse_surf_g_d_0_s_92 = m.lbound[0];
    x->__f2dace_SOA_sw_dn_diffuse_surf_g_d_1_s_93 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_dn_diffuse_surf_g = new std::remove_pointer<
        decltype(x->sw_dn_diffuse_surf_g)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_dn_diffuse_surf_g[i]), s);
    }

  }  // CONCLUDING IF
  read_line(
      s, {"# sw_dn_direct_surf_g"});  // Should contain '# sw_dn_direct_surf_g'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_dn_direct_surf_g_d_0_s_94 = m.size[0];
    x->__f2dace_SA_sw_dn_direct_surf_g_d_1_s_95 = m.size[1];
    x->__f2dace_SOA_sw_dn_direct_surf_g_d_0_s_94 = m.lbound[0];
    x->__f2dace_SOA_sw_dn_direct_surf_g_d_1_s_95 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_dn_direct_surf_g = new std::remove_pointer<
        decltype(x->sw_dn_direct_surf_g)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_dn_direct_surf_g[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s,
            {"# sw_dn_diffuse_surf_clear_g"});  // Should contain '#
                                                // sw_dn_diffuse_surf_clear_g'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_dn_diffuse_surf_clear_g_d_0_s_96 = m.size[0];
    x->__f2dace_SA_sw_dn_diffuse_surf_clear_g_d_1_s_97 = m.size[1];
    x->__f2dace_SOA_sw_dn_diffuse_surf_clear_g_d_0_s_96 = m.lbound[0];
    x->__f2dace_SOA_sw_dn_diffuse_surf_clear_g_d_1_s_97 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_dn_diffuse_surf_clear_g = new std::remove_pointer<
        decltype(x->sw_dn_diffuse_surf_clear_g)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_dn_diffuse_surf_clear_g[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# sw_dn_direct_surf_clear_g"});  // Should contain '#
                                                  // sw_dn_direct_surf_clear_g'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_dn_direct_surf_clear_g_d_0_s_98 = m.size[0];
    x->__f2dace_SA_sw_dn_direct_surf_clear_g_d_1_s_99 = m.size[1];
    x->__f2dace_SOA_sw_dn_direct_surf_clear_g_d_0_s_98 = m.lbound[0];
    x->__f2dace_SOA_sw_dn_direct_surf_clear_g_d_1_s_99 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_dn_direct_surf_clear_g = new std::remove_pointer<
        decltype(x->sw_dn_direct_surf_clear_g)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_dn_direct_surf_clear_g[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# sw_dn_surf_band"});  // Should contain '# sw_dn_surf_band'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_dn_surf_band_d_0_s_100 = m.size[0];
    x->__f2dace_SA_sw_dn_surf_band_d_1_s_101 = m.size[1];
    x->__f2dace_SOA_sw_dn_surf_band_d_0_s_100 = m.lbound[0];
    x->__f2dace_SOA_sw_dn_surf_band_d_1_s_101 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_dn_surf_band =
        new std::remove_pointer<decltype(x->sw_dn_surf_band)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_dn_surf_band[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# sw_dn_direct_surf_band"});  // Should contain '#
                                               // sw_dn_direct_surf_band'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_dn_direct_surf_band_d_0_s_102 = m.size[0];
    x->__f2dace_SA_sw_dn_direct_surf_band_d_1_s_103 = m.size[1];
    x->__f2dace_SOA_sw_dn_direct_surf_band_d_0_s_102 = m.lbound[0];
    x->__f2dace_SOA_sw_dn_direct_surf_band_d_1_s_103 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_dn_direct_surf_band = new std::remove_pointer<
        decltype(x->sw_dn_direct_surf_band)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_dn_direct_surf_band[i]), s);
    }

  }  // CONCLUDING IF
  read_line(
      s,
      {"# sw_dn_surf_clear_band"});  // Should contain '# sw_dn_surf_clear_band'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_dn_surf_clear_band_d_0_s_104 = m.size[0];
    x->__f2dace_SA_sw_dn_surf_clear_band_d_1_s_105 = m.size[1];
    x->__f2dace_SOA_sw_dn_surf_clear_band_d_0_s_104 = m.lbound[0];
    x->__f2dace_SOA_sw_dn_surf_clear_band_d_1_s_105 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_dn_surf_clear_band = new std::remove_pointer<
        decltype(x->sw_dn_surf_clear_band)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_dn_surf_clear_band[i]), s);
    }

  }  // CONCLUDING IF
  read_line(
      s, {"# sw_dn_direct_surf_clear_band"});  // Should contain '#
                                               // sw_dn_direct_surf_clear_band'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_dn_direct_surf_clear_band_d_0_s_106 = m.size[0];
    x->__f2dace_SA_sw_dn_direct_surf_clear_band_d_1_s_107 = m.size[1];
    x->__f2dace_SOA_sw_dn_direct_surf_clear_band_d_0_s_106 = m.lbound[0];
    x->__f2dace_SOA_sw_dn_direct_surf_clear_band_d_1_s_107 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_dn_direct_surf_clear_band = new std::remove_pointer<
        decltype(x->sw_dn_direct_surf_clear_band)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_dn_direct_surf_clear_band[i]), s);
    }

  }  // CONCLUDING IF
}

void deserialize(thermodynamics_type* x, std::istream& s) {
  bool yep;
  array_meta m;
  read_line(s, {"# pressure_hl"});  // Should contain '# pressure_hl'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_pressure_hl_d_0_s_128 = m.size[0];
    x->__f2dace_SA_pressure_hl_d_1_s_129 = m.size[1];
    x->__f2dace_SOA_pressure_hl_d_0_s_128 = m.lbound[0];
    x->__f2dace_SOA_pressure_hl_d_1_s_129 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->pressure_hl =
        new std::remove_pointer<decltype(x->pressure_hl)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->pressure_hl[i]), s);
    }

  }  // CONCLUDING IF
}

void deserialize(single_level_type* x, std::istream& s) {
  bool yep;
  array_meta m;
  read_line(s, {"# sw_albedo"});  // Should contain '# sw_albedo'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_albedo_d_0_s_120 = m.size[0];
    x->__f2dace_SA_sw_albedo_d_1_s_121 = m.size[1];
    x->__f2dace_SOA_sw_albedo_d_0_s_120 = m.lbound[0];
    x->__f2dace_SOA_sw_albedo_d_1_s_121 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_albedo =
        new std::remove_pointer<decltype(x->sw_albedo)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_albedo[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# sw_albedo_direct"});  // Should contain '# sw_albedo_direct'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_sw_albedo_direct_d_0_s_122 = m.size[0];
    x->__f2dace_SA_sw_albedo_direct_d_1_s_123 = m.size[1];
    x->__f2dace_SOA_sw_albedo_direct_d_0_s_122 = m.lbound[0];
    x->__f2dace_SOA_sw_albedo_direct_d_1_s_123 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->sw_albedo_direct = new std::remove_pointer<
        decltype(x->sw_albedo_direct)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->sw_albedo_direct[i]), s);
    }

  }  // CONCLUDING IF
  read_line(s, {"# lw_emissivity"});  // Should contain '# lw_emissivity'

  read_line(s, {"# alloc"});  // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) {  // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_lw_emissivity_d_0_s_124 = m.size[0];
    x->__f2dace_SA_lw_emissivity_d_1_s_125 = m.size[1];
    x->__f2dace_SOA_lw_emissivity_d_0_s_124 = m.lbound[0];
    x->__f2dace_SOA_lw_emissivity_d_1_s_125 = m.lbound[1];
    read_line(s, {"# entries"});  // Should contain '# entries'
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->lw_emissivity =
        new std::remove_pointer<decltype(x->lw_emissivity)>::type[m.volume()];
    for (int i = 0; i < m.volume(); ++i) {
      deserialize(&(x->lw_emissivity[i]), s);
    }

  }  // CONCLUDING IF
}

std::string config_injection(const aerosol_type& x) {
  std::stringstream out;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_od_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_od_sw_d_0_s_61 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_od_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_od_sw_d_0_s_61 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_od_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_od_sw_d_1_s_62 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_od_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_od_sw_d_1_s_62 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_od_sw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_od_sw_d_2_s_63 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_od_sw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_od_sw_d_2_s_63 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_ssa_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ssa_sw_d_0_s_64 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_ssa_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ssa_sw_d_0_s_64 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_ssa_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ssa_sw_d_1_s_65 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_ssa_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ssa_sw_d_1_s_65 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_ssa_sw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ssa_sw_d_2_s_66 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_ssa_sw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ssa_sw_d_2_s_66 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_g_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_g_sw_d_0_s_67 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_g_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_g_sw_d_0_s_67 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_g_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_g_sw_d_1_s_68 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_g_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_g_sw_d_1_s_68 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_g_sw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_g_sw_d_2_s_69 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_g_sw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_g_sw_d_2_s_69 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_od_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_od_lw_d_0_s_70 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_od_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_od_lw_d_0_s_70 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_od_lw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_od_lw_d_1_s_71 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_od_lw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_od_lw_d_1_s_71 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_od_lw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_od_lw_d_2_s_72 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_od_lw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_od_lw_d_2_s_72 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_ssa_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ssa_lw_d_0_s_73 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_ssa_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ssa_lw_d_0_s_73 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_ssa_lw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ssa_lw_d_1_s_74 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_ssa_lw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ssa_lw_d_1_s_74 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SA_ssa_lw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ssa_lw_d_2_s_75 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"__f2dace_SOA_ssa_lw_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ssa_lw_d_2_s_75 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"od_sw_a\", ";
  out << "\"value\": \"" << (x.od_sw ? "true" : "false") << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"ssa_sw_a\", ";
  out << "\"value\": \"" << (x.ssa_sw ? "true" : "false") << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"g_sw_a\", ";
  out << "\"value\": \"" << (x.g_sw ? "true" : "false") << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"od_lw_a\", ";
  out << "\"value\": \"" << (x.od_lw ? "true" : "false") << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_aerosol.aerosol_type\", ";
  out << "\"component\": \"ssa_lw_a\", ";
  out << "\"value\": \"" << (x.ssa_lw ? "true" : "false") << "\"}" << std::endl;
  return out.str();
}

std::string config_injection(const cloud_type& x) {
  std::stringstream out;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_mixing_ratio_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_mixing_ratio_d_0_s_143 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_mixing_ratio_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_mixing_ratio_d_0_s_143 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_mixing_ratio_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_mixing_ratio_d_1_s_144 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_mixing_ratio_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_mixing_ratio_d_1_s_144 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_mixing_ratio_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_mixing_ratio_d_2_s_145 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_mixing_ratio_d_2_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_mixing_ratio_d_2_s_145 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_q_liq_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_q_liq_d_0_s_146 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_q_liq_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_q_liq_d_0_s_146 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_q_liq_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_q_liq_d_1_s_147 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_q_liq_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_q_liq_d_1_s_147 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_q_ice_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_q_ice_d_0_s_148 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_q_ice_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_q_ice_d_0_s_148 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_q_ice_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_q_ice_d_1_s_149 << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_q_ice_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_q_ice_d_1_s_149 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_re_liq_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_re_liq_d_0_s_150 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_re_liq_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_re_liq_d_0_s_150 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_re_liq_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_re_liq_d_1_s_151 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_re_liq_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_re_liq_d_1_s_151 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_re_ice_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_re_ice_d_0_s_152 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_re_ice_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_re_ice_d_0_s_152 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_re_ice_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_re_ice_d_1_s_153 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_re_ice_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_re_ice_d_1_s_153 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_fraction_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_fraction_d_0_s_154 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_fraction_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_fraction_d_0_s_154 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SA_fraction_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_fraction_d_1_s_155 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"__f2dace_SOA_fraction_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_fraction_d_1_s_155 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"mixing_ratio_a\", ";
  out << "\"value\": \"" << (x.mixing_ratio ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud.cloud_type\", ";
  out << "\"component\": \"fraction_a\", ";
  out << "\"value\": \"" << (x.fraction ? "true" : "false") << "\"}"
      << std::endl;
  return out.str();
}

std::string config_injection(const cloud_optics_type& x) {
  std::stringstream out;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SA_liq_coeff_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_liq_coeff_lw_d_0_s_28 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SOA_liq_coeff_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_liq_coeff_lw_d_0_s_28 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SA_liq_coeff_lw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_liq_coeff_lw_d_1_s_29 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SOA_liq_coeff_lw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_liq_coeff_lw_d_1_s_29 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SA_liq_coeff_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_liq_coeff_sw_d_0_s_30 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SOA_liq_coeff_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_liq_coeff_sw_d_0_s_30 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SA_liq_coeff_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_liq_coeff_sw_d_1_s_31 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SOA_liq_coeff_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_liq_coeff_sw_d_1_s_31 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SA_ice_coeff_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ice_coeff_lw_d_0_s_32 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SOA_ice_coeff_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ice_coeff_lw_d_0_s_32 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SA_ice_coeff_lw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ice_coeff_lw_d_1_s_33 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SOA_ice_coeff_lw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ice_coeff_lw_d_1_s_33 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SA_ice_coeff_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ice_coeff_sw_d_0_s_34 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SOA_ice_coeff_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ice_coeff_sw_d_0_s_34 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SA_ice_coeff_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_ice_coeff_sw_d_1_s_35 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"__f2dace_SOA_ice_coeff_sw_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_ice_coeff_sw_d_1_s_35 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"liq_coeff_lw_a\", ";
  out << "\"value\": \"" << (x.liq_coeff_lw ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"liq_coeff_sw_a\", ";
  out << "\"value\": \"" << (x.liq_coeff_sw ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"ice_coeff_lw_a\", ";
  out << "\"value\": \"" << (x.ice_coeff_lw ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_cloud_optics_data.cloud_optics_type\", ";
  out << "\"component\": \"ice_coeff_sw_a\", ";
  out << "\"value\": \"" << (x.ice_coeff_sw ? "true" : "false") << "\"}"
      << std::endl;
  return out.str();
}

std::string config_injection(const gas_type& x) {
  std::stringstream out;

  return out.str();
}

std::string config_injection(const config_type& x) {
  std::stringstream out;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SA_i_emiss_from_band_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_i_emiss_from_band_lw_d_0_s_41 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SOA_i_emiss_from_band_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_i_emiss_from_band_lw_d_0_s_41
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_albedo_weights_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_albedo_weights_d_0_s_42 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_albedo_weights_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_albedo_weights_d_0_s_42 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_albedo_weights_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_albedo_weights_d_1_s_43 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_albedo_weights_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_albedo_weights_d_1_s_43 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SA_i_band_from_reordered_g_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_i_band_from_reordered_g_lw_d_0_s_44
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SOA_i_band_from_reordered_g_lw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_i_band_from_reordered_g_lw_d_0_s_44
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SA_i_band_from_reordered_g_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_i_band_from_reordered_g_sw_d_0_s_45
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"__f2dace_SOA_i_band_from_reordered_g_sw_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_i_band_from_reordered_g_sw_d_0_s_45
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"i_emiss_from_band_lw_a\", ";
  out << "\"value\": \"" << (x.i_emiss_from_band_lw ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"sw_albedo_weights_a\", ";
  out << "\"value\": \"" << (x.sw_albedo_weights ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"i_band_from_reordered_g_lw_a\", ";
  out << "\"value\": \"" << (x.i_band_from_reordered_g_lw ? "true" : "false")
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_config.config_type\", ";
  out << "\"component\": \"i_band_from_reordered_g_sw_a\", ";
  out << "\"value\": \"" << (x.i_band_from_reordered_g_sw ? "true" : "false")
      << "\"}" << std::endl;
  return out.str();
}

std::string config_injection(const flux_type& x) {
  std::stringstream out;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_diffuse_surf_g_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_diffuse_surf_g_d_0_s_92 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_diffuse_surf_g_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_diffuse_surf_g_d_0_s_92
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_diffuse_surf_g_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_diffuse_surf_g_d_1_s_93 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_diffuse_surf_g_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_diffuse_surf_g_d_1_s_93
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_direct_surf_g_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_direct_surf_g_d_0_s_94 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_direct_surf_g_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_direct_surf_g_d_0_s_94 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_direct_surf_g_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_direct_surf_g_d_1_s_95 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_direct_surf_g_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_direct_surf_g_d_1_s_95 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_diffuse_surf_clear_g_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_diffuse_surf_clear_g_d_0_s_96
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_diffuse_surf_clear_g_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_diffuse_surf_clear_g_d_0_s_96
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_diffuse_surf_clear_g_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_diffuse_surf_clear_g_d_1_s_97
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_diffuse_surf_clear_g_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_diffuse_surf_clear_g_d_1_s_97
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_direct_surf_clear_g_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_direct_surf_clear_g_d_0_s_98
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_direct_surf_clear_g_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_direct_surf_clear_g_d_0_s_98
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_direct_surf_clear_g_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_direct_surf_clear_g_d_1_s_99
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_direct_surf_clear_g_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_direct_surf_clear_g_d_1_s_99
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_surf_band_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_surf_band_d_0_s_100 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_surf_band_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_surf_band_d_0_s_100 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_surf_band_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_surf_band_d_1_s_101 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_surf_band_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_surf_band_d_1_s_101 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_direct_surf_band_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_direct_surf_band_d_0_s_102
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_direct_surf_band_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_direct_surf_band_d_0_s_102
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_direct_surf_band_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_direct_surf_band_d_1_s_103
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_direct_surf_band_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_direct_surf_band_d_1_s_103
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_surf_clear_band_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_surf_clear_band_d_0_s_104
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_surf_clear_band_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_surf_clear_band_d_0_s_104
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_surf_clear_band_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_surf_clear_band_d_1_s_105
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_surf_clear_band_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_dn_surf_clear_band_d_1_s_105
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_direct_surf_clear_band_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_direct_surf_clear_band_d_0_s_106
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_direct_surf_clear_band_d_0_s\", ";
  out << "\"value\": \""
      << x.__f2dace_SOA_sw_dn_direct_surf_clear_band_d_0_s_106 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_dn_direct_surf_clear_band_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_dn_direct_surf_clear_band_d_1_s_107
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_dn_direct_surf_clear_band_d_1_s\", ";
  out << "\"value\": \""
      << x.__f2dace_SOA_sw_dn_direct_surf_clear_band_d_1_s_107 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"sw_dn_diffuse_surf_g_a\", ";
  out << "\"value\": \"" << (x.sw_dn_diffuse_surf_g ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"sw_dn_direct_surf_g_a\", ";
  out << "\"value\": \"" << (x.sw_dn_direct_surf_g ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"sw_dn_diffuse_surf_clear_g_a\", ";
  out << "\"value\": \"" << (x.sw_dn_diffuse_surf_clear_g ? "true" : "false")
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"sw_dn_direct_surf_clear_g_a\", ";
  out << "\"value\": \"" << (x.sw_dn_direct_surf_clear_g ? "true" : "false")
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"sw_dn_surf_band_a\", ";
  out << "\"value\": \"" << (x.sw_dn_surf_band ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"sw_dn_direct_surf_band_a\", ";
  out << "\"value\": \"" << (x.sw_dn_direct_surf_band ? "true" : "false")
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"sw_dn_surf_clear_band_a\", ";
  out << "\"value\": \"" << (x.sw_dn_surf_clear_band ? "true" : "false")
      << "\"}" << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_flux.flux_type\", ";
  out << "\"component\": \"sw_dn_direct_surf_clear_band_a\", ";
  out << "\"value\": \"" << (x.sw_dn_direct_surf_clear_band ? "true" : "false")
      << "\"}" << std::endl;
  return out.str();
}

std::string config_injection(const thermodynamics_type& x) {
  std::stringstream out;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_thermodynamics.thermodynamics_type\", ";
  out << "\"component\": \"__f2dace_SA_pressure_hl_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_pressure_hl_d_0_s_128 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_thermodynamics.thermodynamics_type\", ";
  out << "\"component\": \"__f2dace_SOA_pressure_hl_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_pressure_hl_d_0_s_128 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_thermodynamics.thermodynamics_type\", ";
  out << "\"component\": \"__f2dace_SA_pressure_hl_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_pressure_hl_d_1_s_129 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_thermodynamics.thermodynamics_type\", ";
  out << "\"component\": \"__f2dace_SOA_pressure_hl_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_pressure_hl_d_1_s_129 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_thermodynamics.thermodynamics_type\", ";
  out << "\"component\": \"pressure_hl_a\", ";
  out << "\"value\": \"" << (x.pressure_hl ? "true" : "false") << "\"}"
      << std::endl;
  return out.str();
}

std::string config_injection(const single_level_type& x) {
  std::stringstream out;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_albedo_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_albedo_d_0_s_120 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_albedo_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_albedo_d_0_s_120 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_albedo_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_albedo_d_1_s_121 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_albedo_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_albedo_d_1_s_121 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_albedo_direct_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_albedo_direct_d_0_s_122 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_albedo_direct_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_albedo_direct_d_0_s_122 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SA_sw_albedo_direct_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_sw_albedo_direct_d_1_s_123 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SOA_sw_albedo_direct_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_sw_albedo_direct_d_1_s_123 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SA_lw_emissivity_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_lw_emissivity_d_0_s_124 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SOA_lw_emissivity_d_0_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_lw_emissivity_d_0_s_124 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SA_lw_emissivity_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SA_lw_emissivity_d_1_s_125 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"__f2dace_SOA_lw_emissivity_d_1_s\", ";
  out << "\"value\": \"" << x.__f2dace_SOA_lw_emissivity_d_1_s_125 << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"sw_albedo_a\", ";
  out << "\"value\": \"" << (x.sw_albedo ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"sw_albedo_direct_a\", ";
  out << "\"value\": \"" << (x.sw_albedo_direct ? "true" : "false") << "\"}"
      << std::endl;
  out << "{";
  out << "\"type\": \"ConstTypeInjection\", ";
  out << "\"scope\": null, ";
  out << "\"root\": \"radiation_single_level.single_level_type\", ";
  out << "\"component\": \"lw_emissivity_a\", ";
  out << "\"value\": \"" << (x.lw_emissivity ? "true" : "false") << "\"}"
      << std::endl;
  return out.str();
}

}  // namespace serde

#endif  // __DACE_SERDE__