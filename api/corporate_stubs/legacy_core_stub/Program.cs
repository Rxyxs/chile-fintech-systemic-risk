// LEGACY CORE INTEGRATION STUB — NOT a real bank connection.
//
// This simulates the request/response shape of a typical Chilean bank's
// legacy core (SOAP/XML-ish over a synchronous call, account-centric
// modeling) so the Go prediction service in /api/go has a documented target
// to eventually integrate against. There is no real bank system here — this
// is explicitly a stub, not a functional service, and it does not call out
// to anything.
using System;
using System.Collections.Generic;

namespace LegacyCoreStub;

public record AccountHolder(string RutMasked, string FullName, decimal CreditLimitClp);

public record CreditDecisionRequest(string RutMasked, double PdScore, decimal RequestedAmountClp);

public record CreditDecisionResponse(string RutMasked, bool Approved, string Reason);

public interface ILegacyCoreClient
{
    AccountHolder? LookupAccountHolder(string rutMasked);
    CreditDecisionResponse SubmitCreditDecision(CreditDecisionRequest request);
}

// In-memory fake standing in for what would be a SOAP/legacy-core call.
// No network I/O — this is a stub by design, not a mock hiding a bug.
public class InMemoryLegacyCoreClient : ILegacyCoreClient
{
    private readonly Dictionary<string, AccountHolder> _accounts = new()
    {
        ["12.345.678-9"] = new AccountHolder("12.345.678-9", "Titular Simulado A", 3_000_000m),
        ["9.876.543-2"] = new AccountHolder("9.876.543-2", "Titular Simulado B", 1_500_000m),
    };

    public AccountHolder? LookupAccountHolder(string rutMasked) =>
        _accounts.TryGetValue(rutMasked, out var holder) ? holder : null;

    public CreditDecisionResponse SubmitCreditDecision(CreditDecisionRequest request)
    {
        // Threshold mirrors the PD scores actually produced by
        // ml_predictions/train_pd_model.py and served by api/go — this stub
        // is downstream-consistent with the real model output, even though
        // this integration layer itself is simulated.
        bool approved = request.PdScore < 0.15;
        string reason = approved
            ? "PD score below internal risk threshold (simulated policy)"
            : "PD score exceeds internal risk threshold (simulated policy)";
        return new CreditDecisionResponse(request.RutMasked, approved, reason);
    }
}

public static class Program
{
    public static void Main()
    {
        ILegacyCoreClient client = new InMemoryLegacyCoreClient();

        var holder = client.LookupAccountHolder("12.345.678-9");
        Console.WriteLine(holder is not null
            ? $"Found: {holder.FullName}, credit_limit_clp={holder.CreditLimitClp}"
            : "Account holder not found");

        // pd_score below matches the range actually produced by the XGBoost
        // model in this repo (see ml_predictions/reports/metrics.txt)
        var decision = client.SubmitCreditDecision(new CreditDecisionRequest("12.345.678-9", 0.0724, 500_000m));
        Console.WriteLine($"Decision for {decision.RutMasked}: approved={decision.Approved} ({decision.Reason})");

        var decisionRejected = client.SubmitCreditDecision(new CreditDecisionRequest("9.876.543-2", 0.31, 800_000m));
        Console.WriteLine($"Decision for {decisionRejected.RutMasked}: approved={decisionRejected.Approved} ({decisionRejected.Reason})");
    }
}
