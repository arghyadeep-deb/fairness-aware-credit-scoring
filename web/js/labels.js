"use strict";
// Friendly names shared by the form and the SHAP chart. Ranges and options still come from /meta/options.
const FieldLabels = (() => {
  const form = {
    age_years: "Age (years)", cnt_children: "Number of children", cnt_fam_members: "Family members",
    amt_income_total: "Total income", amt_credit: "Credit amount", amt_annuity: "Loan annuity",
    amt_goods_price: "Goods price", region_population_relative: "Region population (relative)",
    years_employed: "Years employed", name_family_status: "Family status", name_education_type: "Education",
    name_income_type: "Income type", name_housing_type: "Housing type", occupation_type: "Occupation",
    organization_type: "Organization type", flag_own_car: "Owns a car", flag_own_realty: "Owns real estate",
    ext_source_1: "External score 1", ext_source_2: "External score 2", ext_source_3: "External score 3",
  };
  const derived = {
    CREDIT_INCOME_RATIO: "Credit-to-income ratio", ANNUITY_INCOME_RATIO: "Annuity-to-income ratio",
    CREDIT_TERM: "Credit term (credit / annuity)",
  };
  const feature = (raw) => derived[raw] || form[raw.toLowerCase()] || raw;
  function shapLabel(item) {
    const name = feature(item.feature);
    if (item.missing) return name + ": not provided";
    if (item.value === null || item.value === undefined) return name;
    if (item.feature.startsWith("FLAG_")) return name + " = " + (Number(item.value) === 1 ? "Yes" : "No");
    return name + " = " + item.value;
  }
  return { form, feature, shapLabel };
})();
