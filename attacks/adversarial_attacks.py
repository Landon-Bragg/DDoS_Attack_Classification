from torchattacks import FGSM, PGD

# Returns adversarial attack modules
get_fgsm = lambda model, eps: FGSM(model, eps=eps)
get_pgd = lambda model, eps, alpha, steps: PGD(model, eps=eps, alpha=alpha, steps=steps)